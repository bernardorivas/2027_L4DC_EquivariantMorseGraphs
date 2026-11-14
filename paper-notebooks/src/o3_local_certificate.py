"""Arb bounds for the saved real-valued scalar EMLP quotient map.

The export is the two scalar EMLP blocks (bilinear parts identically zero).
Weights are exact binary64 numbers, arithmetic/transcendentals are enclosed by
Arb. This certifies inequalities of the real network, not hardware roundoff.
"""
from pathlib import Path
import sys,json,hashlib,time,math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'build/verification-deps'))
from flint import arb,arb_mat,ctx
ctx.prec=128
BASE=ROOT/'build/o3-certification'

def exact(x):
    n,d=float(x).as_integer_ratio();return arb(n)/arb(d)

def interval(a,b):return exact(a).union(exact(b))

def bounds(x):return [math.nextafter(float(x.lower()),-math.inf),math.nextafter(float(x.upper()),math.inf)]

class Network:
    def __init__(self):
        self.record=json.loads((BASE/'network.json').read_text())
        self.layers=[(arb_mat([[exact(x) for x in row] for row in l['W']]),arb_mat([[exact(x)] for x in l['b']]),l['activation']) for l in self.record['layers']]
    def __call__(self,box,derivatives=False):
        q=[interval(box[0],box[2]),interval(box[1],box[3])];x=arb_mat([[v] for v in q]);jac=arb_mat([[1,0],[0,1]])
        for W,b,activation in self.layers:
            x=W*x+b
            if derivatives:jac=W*jac
            if activation=='swish':
                for i in range(x.nrows()):
                    v=x[i,0];sig=1/(1+(-v).exp())
                    if derivatives:
                        slope=sig+v*sig*(1-sig)
                        for j in range(2):jac[i,j]*=slope
                    x[i,0]=v*sig
        gains=[];values=[];J=[]
        for i in range(2):
            t=(x[i,0]/3).tanh();g=(3*t).exp();gains.append(g);values.append(q[i]*g)
            if derivatives:J.append([g*(int(i==j)+q[i]*(1-t*t)*jac[i,j]) for j in range(2)])
        return gains,values,J


def prove_cover(network,box,kind,max_depth=16):
    stack=[(list(box),0)];accepted=[];count=0
    while stack:
        b,depth=stack.pop();count+=1
        if count>20000:raise RuntimeError('Subdivision cap')
        gains,values,J=network(b,derivatives=kind.startswith('axis'))
        if kind=='origin':tests=[gains[0]>1,gains[1]>1];watched=gains
        elif kind=='saddle':tests=[gains[0]>1];watched=[gains[0]]
        elif kind=='sink':tests=[gains[1]<1];watched=[gains[1]]
        elif kind=='axis_a':tests=[J[0][0]>0,J[0][0]<1];watched=[J[0][0]]
        elif kind=='axis_r':tests=[J[1][1]>0,J[1][1]<1];watched=[J[1][1]]
        else:raise ValueError(kind)
        if all(tests):accepted.append([bounds(v) for v in watched]);continue
        if depth>=max_depth:raise RuntimeError(f'Unresolved {kind} {b} {[bounds(v) for v in watched]}')
        axis=max(range(2),key=lambda i:b[i+2]-b[i]);mid=(b[axis]+b[axis+2])/2
        left=b.copy();right=b.copy();left[axis+2]=mid;right[axis]=mid
        stack.extend([(left,depth+1),(right,depth+1)])
    return dict(condition=kind,box=box,subboxes=len(accepted),evaluations=count,
                bounds=[[min(v[i][0] for v in accepted),max(v[i][1] for v in accepted)] for i in range(len(accepted[0]))],verified=True)


def run(out=None):
    out=Path(out) if out else BASE
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();network=Network()
    study=ROOT/'build/o3-repair/cmgdb-001/study'
    agg=json.loads((ROOT/'build/o3-aggregation/result-256.json').read_text())
    comp=json.loads((study/'comparison.json').read_text());native=comp['resolutions'][-1]['rectangle']['nodes'];nodes={r['id']:r for r in native}
    roots=comp['point_checks']['roots'];records=[]
    for root in roots:
        group=agg['groups'][str(root['id'])]
        lo=[min(nodes[k]['lower'][i] for k in group) for i in range(2)]
        hi=[max(nodes[k]['upper'][i] for k in group) for i in range(2)]
        # Widen a little so the root and aggregated cells lie in the checked block.
        lo=[lo[0]-1e-6,max(0.,lo[1]-1e-6)];hi=[x+1e-6 for x in hi];box=lo+hi
        role='C' if root['q'][1]>0 else ('O' if root['q'][0]==0 else ('A-' if root['q'][0]<0 else 'A+'))
        kind='origin' if role=='O' else ('saddle' if role=='C' else 'sink')
        checks=[prove_cover(network,box,kind)]
        bracket=None
        if role!='O':
            axis=1 if role=='C' else 0
            axisbox=[0.,0.,0.,0.];axisbox[axis]=lo[axis];axisbox[axis+2]=hi[axis]
            checks.append(prove_cover(network,axisbox,'axis_r' if axis else 'axis_a'))
            signs=[]
            for x in (lo[axis],hi[axis]):
                q=[0.,0.];q[axis]=x
                value=network(q+q)[1][axis]-exact(x);signs.append(value)
            assert signs[0]>0 and signs[1]<0
            bracket=dict(coordinate=axis,interval=[lo[axis],hi[axis]],endpoint_displacements=list(map(bounds,signs)),unique=True)
        records.append(dict(role=role,group=group,box=box,checks=checks,fixed_point_bracket=bracket))
        print(role,json.dumps(records[-1]),flush=True)
    # Independent JAX comparison of the interval interpretation at stored points.
    maxerr=0.;jacerr=0.
    for q,expected,expected_jac in zip(network.record['samples'][:20],network.record['jax_outputs'][:20],network.record['jax_jacobians'][:20]):
        _,values,J=network(q+q,derivatives=True)
        maxerr=max(maxerr,max(abs(float(values[i].mid())-expected[i]) for i in range(2)))
        jacerr=max(jacerr,max(abs(float(J[i][j].mid())-expected_jac[i][j]) for i in range(2) for j in range(2)))
    assert maxerr<1e-12 and jacerr<1e-11
    result=dict(precision_bits=ctx.prec,library='python-flint 0.8.0 / Arb',checkpoint_sha256=network.record['checkpoint_sha256'],
       regions=records,all_local_inequalities_verified=True,jax_comparison_max_error=maxerr,jax_autodiff_max_error=jacerr,seconds=time.monotonic()-start,
       mathematical_scope='Real-valued saved scalar network and its specified direction-preserving O(3) lift in latent coordinates.',
       not_certified=['Global encoder injectivity/local conjugacy to physical coordinates','Uniform data-model error','Global completeness of sampled Morse graph','Hardware floating-point iteration errors'])
    (out/'local-certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args();run(args.output)
