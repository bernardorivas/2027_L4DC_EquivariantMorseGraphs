"""Contract the published EMLP bilinear operator without its dense matrix."""
import inspect
import numpy as np
import jax.numpy as jnp
import objax
from emlp.nn.objax import gated


class BlockBilinear(objax.Module):
    def __init__(self,native,rep):
        self.w=native.w
        fun=native.weight_proj
        while hasattr(fun,'__wrapped__'):fun=fun.__wrapped__
        meta=inspect.getclosurevars(fun).nonlocals
        nin,nout=meta['mat_shape'][1],meta['mat_shape'][0]
        assert nin==nout==rep.size()
        groups=list(rep.as_dict(np.arange(rep.size())).items())
        offsets={};co=po=0
        for typ,mult in meta['W_multiplicities'].items():
            dim=typ.size()
            if typ in meta['x_multiplicities']:
                ns=min(meta['x_multiplicities'][typ],dim)
                offsets[typ]=(co,po,ns)
                po+=mult*ns
            co+=mult*dim
        self.parts=[];order=[]
        for a,oi in groups:
            da=oi.shape[1];pieces=[];order.extend(oi.ravel().tolist())
            for b,ii in groups:
                db=ii.shape[1];typ=(a*b.T).canonicalize()[0]
                if typ not in offsets:continue
                c0,p0,ns=offsets[typ];dim=da*db
                canonical=meta['inv_perm'][oi[:,None,:,None]*nin+ii[None,:,None,:]]-c0
                rr=canonical//dim;components=canonical%dim
                assert np.all(rr==rr[:,:,0,0,None,None])
                assert np.all(components==np.arange(dim).reshape(1,1,da,db))
                theta=p0+rr[:,:,0,0,None]*ns+np.arange(ns)
                bids=meta['reduced_indices_dict'][typ]
                pieces.append((jnp.array(ii),jnp.array(theta),jnp.array(bids),ns,da,db))
            self.parts.append((len(oi),da,pieces))
        self.inverse=jnp.array(np.argsort(order))

    def __call__(self,x,training=True):
        blocks=[]
        for copies,da,pieces in self.parts:
            y=jnp.zeros((x.shape[0],copies,da),dtype=x.dtype)
            for ii,theta,bids,ns,da,db in pieces:
                selected=x[:,bids].reshape(x.shape[0],ns,da,db)
                products=jnp.einsum('bsoi,bni->bsno',selected,x[:,ii])
                y=y+.1*jnp.einsum('ans,bsno->bao',self.w.value[theta],products)
            blocks.append(y.reshape(x.shape[0],-1))
        return jnp.concatenate(blocks,axis=-1)[:,self.inverse]


class ProjectedLinear(objax.Module):
    def __init__(self,native):
        for name,value in native.__dict__.items():
            if isinstance(value,objax.TrainVar):setattr(self,name,value)

    def __call__(self,x):
        return x @ self.w.value.T+self.b.value


def install(model):
    variables=list(model.vars().values());locations={id(v):i for i,v in enumerate(variables)}
    projectors=[]
    for layer in model.network:
        if hasattr(layer,'bilinear'):
            layer.bilinear=BlockBilinear(layer.bilinear,gated(layer.nonlinearity.rep))
            linear=layer.linear
            projectors.append((locations[id(linear.w)],locations[id(linear.b)],linear.Pw,linear.Pb))
            layer.linear=ProjectedLinear(linear)
        else:
            linear=layer
            projectors.append((locations[id(linear.w)],locations[id(linear.b)],linear.Pw,linear.Pb))
    model.network[-1]=ProjectedLinear(model.network[-1])
    assert [id(v) for v in model.vars().values()]==[id(v) for v in variables]
    return projectors
