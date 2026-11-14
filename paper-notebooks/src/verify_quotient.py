"""Re-evaluate saved direct-quotient checkpoints and reproduce their CMGDB graphs."""
from pathlib import Path
import argparse, json, os, subprocess, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def verify_worker(fit):
    from quotient_runtime import load_quotient
    sys.path.insert(0,str(ROOT/'vendor/latent-equivariant/code/examples'))
    import systems
    cfg,E,D,F=load_quotient(fit)
    result=json.loads((fit/'result.json').read_text())
    data=np.load(fit/'evaluation.npz')
    sphere=cfg['system']=='sphere'; dim=2 if sphere else 3
    raw=data['raw_states'];q=np.asarray(E(raw))
    assert raw.shape[1]==dim+1 and q.shape[1]==(1 if sphere else 2)
    assert result['encoder_input']=='raw state' and not result['synthetic_observations']
    rng=np.random.default_rng(4107)
    errors=[]
    for _ in range(4):
        orthogonal=np.linalg.qr(rng.normal(size=(dim,dim)))[0]
        for sign in (-1,1):
            matrix=orthogonal.copy();matrix[:,0]*=sign
            transformed=raw.copy()
            if sphere:transformed[:,:2]=raw[:,:2]@matrix.T
            else:transformed[:,1:]=raw[:,1:]@matrix.T
            errors.append(float(np.max(np.abs(np.asarray(E(transformed))-q))))
    assert max(errors)<2e-6
    rec=np.asarray(D(q))
    np.testing.assert_allclose(rec,data['decoded_quotient'],rtol=0,atol=1e-12)
    def orbit_rms(decoded,state):
        if sphere:
            h=decoded[:,0];z=state[:,2]
            squared=(h-z)**2+(np.sqrt(np.maximum(1e-15,1-h*h))-np.sqrt(np.maximum(0,1-z*z)))**2
            return float(np.sqrt(squared.mean()))
        squared=(decoded[:,0]-state[:,0])**2+(decoded[:,1]-np.linalg.norm(state[:,1:],axis=1))**2
        return float(np.sqrt(squared.mean())/1.5)
    metrics={'reconstruction_orbit_rms':orbit_rms(rec,raw)}
    next_state=systems.step(cfg['system'],raw,corotating=sphere)
    metrics['prediction_orbit_rms']=orbit_rms(np.asarray(D(F(q))),next_state)
    state,prediction=raw.copy(),q.copy()
    for _ in range(cfg['horizon']):
        state=systems.step(cfg['system'],state,corotating=sphere)
        prediction=np.asarray(F(prediction))
    metrics['rollout_orbit_rms']=orbit_rms(np.asarray(D(prediction)),state)
    for key,value in metrics.items():
        np.testing.assert_allclose(value,result['metrics'][key],atol=1e-12,rtol=0)
    # Recompute selected original cell hulls using the reloaded neural map.
    boxes=data['boxes'];qdim=q.shape[1]
    selected=np.unique(np.r_[0,len(boxes)-1,rng.integers(0,len(boxes),62)])
    fractions=np.linspace(0,1,5)[:,None] if sphere else np.array([(a,b) for a in np.linspace(0,1,5) for b in np.linspace(0,1,5)])
    chosen=boxes[selected]
    sample=chosen[:,None,:qdim]+fractions[None,:,:]*(chosen[:,None,qdim:]-chosen[:,None,:qdim])
    image=sample.reshape(-1,qdim)
    for _ in range(result['quotient_iterations']):image=np.asarray(F(image))
    image=image.reshape(len(chosen),len(fractions),qdim)
    hulls=np.column_stack((image.min(1)-1e-9,image.max(1)+1e-9))
    if sphere:hulls=np.clip(hulls,-1,1)
    else:hulls[:,1]=np.maximum(hulls[:,1],0)
    discrepancy=float(np.max(np.abs(hulls-data['image_boxes'][selected])))
    assert discrepancy<1e-11,discrepancy
    return dict(raw_dimension=raw.shape[1],quotient_dimension=q.shape[1],test_states=len(raw),
        invariance_error=max(errors),orbit_metrics=metrics,checked_cell_hulls=len(selected),
        hull_reload_error=discrepancy,checkpoint_sha256=result['checkpoint_sha256'])


def verify(fits,out):
    sys.path.insert(0,str(ROOT/'src'))
    from notebook_support import workspace
    from example_plots import learned_cmgdb
    workspace()
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for fit in fits:
        result=json.loads((fit/'result.json').read_text())
        system=result['config']['system']
        env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR=str(ROOT/'build/matplotlib'))
        call=subprocess.run([str(ROOT/'.venv-emlp/bin/python'),'-B',__file__,'--worker','--fits',str(fit),
                             '--out',str(out/f'{system}-checkpoint.json')],env=env,text=True,capture_output=True)
        if call.returncode:raise RuntimeError(call.stdout+call.stderr)
        previous=json.loads((fit/'cmgdb.json').read_text())
        current=learned_cmgdb(fit)
        assert current==previous
        (out/f'{system}-cmgdb.json').write_text(json.dumps(current,indent=2)+'\n')
        row=json.loads((out/f'{system}-checkpoint.json').read_text())
        row.update(system=system,cmgdb_native_nodes=current['native_nodes'],cmgdb_reproduced=True,
                   outgoing_image_cells=current['outgoing_image_cells'])
        rows.append(row)
    (out/'verification.json').write_text(json.dumps(dict(certified=False,results=rows),indent=2)+'\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fits',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:args.out.write_text(json.dumps(verify_worker(args.fits[0]),indent=2)+'\n')
    else:verify(args.fits,args.out)
