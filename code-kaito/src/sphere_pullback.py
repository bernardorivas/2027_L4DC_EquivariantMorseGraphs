"""Numerical preimages of native Morse cells under the actual sphere encoder."""
from pathlib import Path
import json,os,re,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def _preimages(fit,study,out):
    from scipy.optimize import brentq
    from quotient_runtime import load_quotient
    _,encode,_,_=load_quotient(fit)
    def E(z):
        return float(np.asarray(encode(np.array([[np.sqrt(max(0,1-z*z)),0.,z]])))[0,0])
    z=np.linspace(-1,1,8193)
    x=np.column_stack((np.sqrt(1-z*z),np.zeros_like(z),z))
    q=np.asarray(encode(x))[:,0]
    if not (np.all(np.diff(q)>0) or np.all(np.diff(q)<0)):
        raise ValueError('This inverse plot requires a monotone height encoder.')
    def inverse(u):
        if abs(u-E(-1))<1e-10:return -1.
        if abs(u-E(1))<1e-10:return 1.
        return brentq(lambda h:E(h)-u,-1,1,xtol=1e-13)
    rows=[]
    for lo,hi,node in np.loadtxt(study/'rectangle-128-cells.csv',delimiter=',',ndmin=2):
        ends=[inverse(lo),inverse(hi)]
        rows.append(dict(node=int(node),q=[lo,hi],z=sorted(ends),residual=max(abs(E(ends[0])-lo),abs(E(ends[1])-hi))))
    # Independent longitude check: the raw-state encoder has the same value on each orbit.
    phi=np.linspace(0,2*np.pi,17);zz=np.linspace(-.99,.99,23)
    xx=np.array([[np.sqrt(1-h*h)*np.cos(t),np.sqrt(1-h*h)*np.sin(t),h] for h in zz for t in phi])
    values=np.asarray(encode(xx)).reshape(len(zz),len(phi))
    result=dict(checkpoint_sha256=json.loads((fit/'result.json').read_text())['checkpoint_sha256'],
        cells=rows,max_inverse_residual=max(r['residual'] for r in rows),
        longitude_error=float(np.max(np.ptp(values,axis=1))),
        interpretation='Numerical encoder preimages of native Morse cells; not attraction regions or certified enclosures.')
    out.write_text(json.dumps(result,indent=2)+'\n')


def draw_transparent_sphere(ax, rows, palette):
    """Orthographic front/rear hemispheres; QuTiP-inspired translucent style.

    Screen coordinates (u,v) have camera depth +/-sqrt(1-u²-v²).
    Physical height is v*cos(elevation) + depth*sin(elevation).
    """
    from matplotlib.patches import Circle
    t=np.linspace(-1,1,901)
    xx,yy=np.meshgrid(t,t);rr=xx*xx+yy*yy
    elev=np.deg2rad(12)
    depth=np.sqrt(np.maximum(0,1-rr))
    for sign,alpha,layer in [(-1,.24,1),(1,.85,3)]:
        height=np.ma.array(yy*np.cos(elev)+sign*depth*np.sin(elev),mask=rr>1)
        if sign==1:
            ax.add_patch(Circle((0,0),1,facecolor='#e6e9ed',edgecolor='none',alpha=.20,zorder=2))
        for row in rows:
            band_alpha=.13 if sign<0 and row['z'][0]<=-1 else alpha
            ax.contourf(xx,yy,height,levels=row['z'],colors=[palette[row['node']]],alpha=band_alpha,zorder=layer)
    ax.add_patch(Circle((0,0),1,facecolor='none',edgecolor='.65',lw=.7,zorder=4))


def show_sphere_pullback(fit,study):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from IPython.display import display,HTML
    folder=ROOT/study['study'];out=folder/'sphere-pullback.json'
    checkpoint=json.loads((fit/'result.json').read_text())['checkpoint_sha256']
    if not out.exists() or json.loads(out.read_text())['checkpoint_sha256']!=checkpoint:
        env=dict(os.environ,JAX_ENABLE_X64='true',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
        process=subprocess.run([str(ROOT/'.venv-emlp/bin/python'),__file__,str(fit),str(folder),str(out)],env=env,capture_output=True,text=True)
        if process.returncode:raise RuntimeError('Sphere preimage evaluation failed:\n'+process.stderr[-4000:])
    evidence=json.loads(out.read_text())
    exact=ROOT/'vendor/latent-equivariant/code/examples/data/sphere/exact-K128'
    explicit=[dict(z=[lo,hi],node=int(k)) for lo,hi,k in np.loadtxt(exact/'morse-sets.csv',delimiter=',',ndmin=2)]
    def colors(path):
        return {int(k):v for k,v in re.findall(r'(\d+) \[.*?fillcolor="(#[0-9a-fA-F]+)"',path.read_text())}
    elev=np.deg2rad(12)
    with plt.rc_context({'font.family':'serif','mathtext.fontset':'cm','font.size':13}):
        fig,axes=plt.subplots(1,2,figsize=(8,4.1),layout='constrained')
        for ax,rows,palette,title in zip(axes,[explicit,evidence['cells']],
                [colors(exact/'morse-graph.dot'),colors(folder/'rectangle-128.dot')],['Analytical','Joint EMLP']):
            draw_transparent_sphere(ax,rows,palette)
            for sign,label in [(1,'$N$'),(-1,'$S$')]:
                ax.plot(0,sign*np.cos(elev),marker='o',ms=3,linestyle='none',
                        markerfacecolor='none' if sign<0 else 'k',markeredgecolor='k',
                        markeredgewidth=.6,alpha=.35 if sign<0 else 1,zorder=1 if sign<0 else 5)
                ax.text(0,sign*1.06,label,ha='center',va='center',fontsize=11)
            ax.set(aspect='equal',xlim=(-1.1,1.1),ylim=(-1.13,1.13),title=title);ax.axis('off')
        for ext in ('png','pdf','svg'):fig.savefig(folder/f'sphere-pullback.{ext}',dpi=220,bbox_inches='tight')
        display(fig);plt.close(fig)
    display(HTML('<p>Preimages of the 128-cell native Morse sets. Colors match each native graph; bands are finite CMGDB cells, not basins of attraction.</p>'))
    return evidence

if __name__=='__main__':_preimages(*map(Path,sys.argv[1:]))
