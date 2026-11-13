"""Read-only checkpoint diagnostics, kept outside the example notebook."""
from pathlib import Path
import sys, json, random
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import objax
from emlp.groups import O
from emlp.reps import T,V
from emlp.nn.objax import EMLP, Standardize
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/latent-equivariant/code/examples'))
import systems


def load(run):
    cfg=json.loads((run/'config.json').read_text())
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); objax.random.DEFAULT_GENERATOR.seed(cfg['seed'])
    obs,lat=3*T(0)+2*V,T(0)+V
    models=objax.ModuleList([EMLP(a,b,group=O(3),ch=cfg['width'],num_layers=cfg['layers']) for a,b in ((obs,lat),(lat,obs),(lat,lat))])
    objax.io.load_var_collection(str(run/'models.npz'),models.vars())
    with np.load(run/'evaluation.npz') as d:
        center,scale,lc,ls=[jnp.asarray(d[k]) for k in ('observation_center','observation_scale','latent_center','latent_scale')]
    fn=Standardize(models[2],(lc,ls,lc,ls))
    E=objax.Jit(lambda y:models[0]((y-center)/scale),models[0].vars())
    D=objax.Jit(lambda z:models[1](z)*scale+center,models[1].vars())
    f=objax.Jit(lambda z:fn(z,training=False),models[2].vars())
    return cfg,E,D,f,np.asarray(scale),np.asarray(ls)


def audit(run, seed=2001, count=512, horizon=10):
    cfg,E,D,f,scale,ls=load(run)
    states=systems.trajectories('o3',systems.sample_states('o3',count,seed=seed),intervals=horizon)
    y=systems.observe('o3',states)
    enc=np.asarray(E(y.reshape(-1,9))).reshape(count,horizon+1,4)
    def rms(pred,target):
        pred=np.asarray(pred)
        if not np.isfinite(pred).all(): return None
        with np.errstate(over='ignore',invalid='ignore'):
            value=float(np.sqrt(np.mean((pred-target)**2)/np.mean(target**2)))
        return value if np.isfinite(value) else None
    rec=np.asarray(D(enc.reshape(-1,4))).reshape(y.shape)
    pred=np.asarray(D(f(enc[:,:-1].reshape(-1,4)))).reshape(count,horizon,9)
    rows=[]
    for name,part in [('scalar',slice(0,3)),('vector',slice(3,9)),('all',slice(None))]:
        rows.append(dict(block=name,reconstruction=rms(rec[:,:,part],y[:,:,part]),initial_reconstruction=rms(rec[:,0,part],y[:,0,part]),first_step=rms(pred[:,0,part],y[:,1,part]),all_pairs=rms(pred[:,:,part],y[:,1:,part])))
    free=enc[:,0].copy(); rollout=[]
    for k in range(1,horizon+1):
        free=np.asarray(f(free)); finite=np.isfinite(free).all(axis=1)
        rollout.append(dict(step=k,finite_count=int(finite.sum()),max_norm=float(np.linalg.norm(free[finite],axis=1).max()) if finite.any() and np.max(np.abs(free[finite]))<1e150 else None,relative_rms=rms(D(free),y[:,k])))
    def qencode(q):
        z=np.asarray(E(systems.observe('o3',np.column_stack((q,np.zeros((len(q),2)))))))
        return np.column_stack((z[:,0],np.linalg.norm(z[:,1:],axis=1)))
    aa,rr=np.meshgrid(np.linspace(-1.49,1.49,101),np.linspace(.01,1.49,61))
    q=np.column_stack((aa.ravel(),rr.ravel())); h=1e-5
    jac=np.stack([(qencode(q+np.eye(2)[i]*h)-qencode(q-np.eye(2)[i]*h))/(2*h) for i in range(2)],axis=-1)
    det=np.linalg.det(jac)
    # Search for distinct physical orbits with the same learned quotient coordinate.
    from scipy.spatial import cKDTree
    from scipy.optimize import least_squares
    mapped=qencode(q); units=ls[:2]
    distances,neighbors=cKDTree(mapped/units).query(mapped/units,k=16)
    separated=np.linalg.norm(q[:,None,:]-q[neighbors],axis=2)>.3
    candidates=np.where(separated,distances,np.inf)
    witness=None
    for flat_index in np.argsort(candidates.ravel())[:20]:
        i,j=np.unravel_index(flat_index,candidates.shape)
        if not np.isfinite(candidates[i,j]): break
        target=mapped[i]
        solution=least_squares(lambda x:(qencode(x[None,:])[0]-target)/units,q[neighbors[i,j]],bounds=([-1.5,0],[1.5,1.5]),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100)
        if np.linalg.norm(solution.fun)<1e-8 and np.linalg.norm(solution.x-q[i])>.3:
            pair=np.stack((q[i],solution.x))
            nxt=systems.step('o3',np.column_stack((pair,np.zeros((2,2)))))
            next_q=np.column_stack((nxt[:,0],np.linalg.norm(nxt[:,1:],axis=1)))
            witness=dict(physical_quotients=pair.tolist(),scaled_encoder_separation=float(np.linalg.norm(solution.fun)),
                         encoded_successor_separation=float(np.linalg.norm((qencode(next_q)[0]-qencode(next_q)[1])/units)))
            break
    eq=np.array([[-1.,0,0,0],[1.,0,0,0],[0.,0,0,0],[0.,1,0,0]])
    ze=E(systems.observe('o3',eq)); residual=np.linalg.norm((np.asarray(f(ze))-np.asarray(ze))/ls,axis=1)
    record=dict(run=str(run.relative_to(ROOT)),seed=seed,count=count,by_block=rows,rollout=rollout,
                sampled_distribution=dict(initial_near_sink_fraction=float(np.mean((np.abs(np.abs(states[:,0,0])-1)<.1)&(np.linalg.norm(states[:,0,1:],axis=1)<.1))),all_times_near_sink_fraction=float(np.mean((np.abs(np.abs(states[:,:,0])-1)<.1)&(np.linalg.norm(states[:,:,1:],axis=2)<.1)))),
                quotient_jacobian=dict(positive_fraction=float(np.mean(det>1e-5)),negative_fraction=float(np.mean(det< -1e-5)),minimum_singular_value=float(np.linalg.svd(jac,compute_uv=False)[:,-1].min())),
                quotient_collision=witness,
                encoded_equilibrium_residual=dict(zip(('sink_minus','sink_plus','origin','saddle_sphere'),residual.tolist())))
    (run/f'diagnostics-{seed}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2))
    return record

if __name__=='__main__': audit(Path(sys.argv[1]).resolve(),seed=int(sys.argv[2]) if len(sys.argv)>2 else 2001)
