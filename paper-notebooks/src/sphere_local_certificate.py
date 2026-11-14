"""Uniform Arb local checks for the saved sphere time-one map and its circle lift."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from o3_local_certificate import arb,arb_mat,ctx,exact,interval,bounds
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'build/sphere-certification'

class Network:
    def __init__(self):
        self.record=json.loads((BASE/'network.json').read_text())
        self.layers=[(arb_mat([[exact(x) for x in row] for row in l['W']]),arb_mat([[exact(x)] for x in l['b']]),l['activation']) for l in self.record['layers']]
    def __call__(self,lo,hi):
        q=interval(lo,hi);x=arb_mat([[q]]);J=arb_mat([[1]])
        for W,b,act in self.layers:
            x=W*x+b;J=W*J
            if act=='swish':
                for i in range(x.nrows()):
                    v=x[i,0];sig=1/(1+(-v).exp())
                    J[i,0]*=sig+v*sig*(1-sig);x[i,0]=v*sig
        n=x[0,0];inner=(n/3).tanh();t=(3*inner).tanh()
        tp=(1-t*t)*(1-inner*inner)*J[0,0]
        f=(q+t)/(1+q*t)
        derivative=((1-t*t)+(1-q*q)*tp)/((1+q*t)**2)
        return n,f,derivative

def cover(net,lo,hi,attractor):
    pending=[(lo,hi,0)];accepted=[]
    while pending:
        a,b,depth=pending.pop();_,_,d=net(a,b)
        good=d>0 if attractor is None else ((d>0 and d<1) if attractor else d>1)
        if good:accepted.append(bounds(d));continue
        if depth>=18 or len(pending)+len(accepted)>20000:raise RuntimeError(f'Unresolved derivative on {(a,b)}: {bounds(d)}')
        c=(a+b)/2;pending.extend([(a,c,depth+1),(c,b,depth+1)])
    return dict(subintervals=len(accepted),derivative_bounds=[min(b[0] for b in accepted),max(b[1] for b in accepted)])

def run():
    start=time.monotonic();net=Network()
    folder=ROOT/'build/sphere-sampling/uniform-4096-cmgdb-001/study'
    record=json.loads((folder/'comparison.json').read_text())
    fit=ROOT/record['fit'];sha=hashlib.sha256((fit/'models.npz').read_bytes()).hexdigest()
    assert sha==net.record['checkpoint_sha256']
    agg=record['resolutions'][0]['aggregate'];cells=np.loadtxt(folder/'rectangle-128-cells.csv',delimiter=',',ndmin=2)
    names={0:'S',1:'A-',2:'C',3:'A+',4:'N'};regions=[]
    for k,name in names.items():
        selected=cells[np.isin(cells[:,2],agg['groups'][str(k)])]
        a=max(-1.,float(selected[:,0].min())-1e-6);b=min(1.,float(selected[:,1].max())+1e-6)
        check=cover(net,a,b,k in (1,3))
        na=net(a,a)[0];nb=net(b,b)[0]
        if k==0:assert na>0 and nb>0
        elif k==4:assert na<0 and nb<0
        elif k in (1,3):assert na>0 and nb<0
        else:assert na<0 and nb>0
        regions.append(dict(role=name,interval=[a,b],native_nodes=agg['groups'][str(k)],**check,
                            endpoint_network_bounds=[bounds(na),bounds(nb)],unique_invariant_orbit=True))
        print(regions[-1],flush=True)
    monotonicity=cover(net,-1.,1.,None)
    value_error=0.;derivative_error=0.
    for q,v,j in zip(net.record['samples'][:30],net.record['jax_outputs'][:30],net.record['jax_jacobians'][:30]):
        _,f,d=net(q[0],q[0]);value_error=max(value_error,abs(float(f.mid())-v[0]));derivative_error=max(derivative_error,abs(float(d.mid())-j[0][0]))
    assert value_error<1e-12 and derivative_error<1e-11
    answer=dict(checkpoint_sha256=sha,network_sha256=hashlib.sha256((BASE/'network.json').read_bytes()).hexdigest(),
        global_monotonicity=monotonicity,precision_bits=ctx.prec,library='python-flint 0.8.0 / Arb',regions=regions,all_local_inequalities_verified=True,
        value_parity_error=value_error,jax_derivative_parity_error=derivative_error,seconds=time.monotonic()-start,
        time_step=1,cmgdb_iterations=10,fiber_map='theta -> theta + 0.6 per time-one step; degree +1; endpoints collapsed',
        scope='Local indices for the real saved interval network lifted to a sphere in latent height q, and its tenth iterate.',
        not_certified=['Encoder conjugacy to physical sphere','Uniform error relative to true ODE','Global completeness or connections of sampled Morse graph','Hardware roundoff iteration'])
    (BASE/'local-certificate.json').write_text(json.dumps(answer,indent=2)+'\n')
    return answer
if __name__=='__main__':run()
