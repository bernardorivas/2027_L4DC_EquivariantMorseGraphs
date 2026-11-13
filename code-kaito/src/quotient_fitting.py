"""Invariant EMLP encoder, quotient dynamics, and orbit reconstruction.

No synthetic observations or full-state latent autoencoder. Direct mode learns
invariant coordinates from raw states. O(3) gain mode explicitly multiplies the
known quotient (a, norm(v)) by learned positive corrections; its invariant prior
is recorded in the run configuration, result, and notebook.
"""
from pathlib import Path
import hashlib, json, os, subprocess, sys, tempfile, time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_quotient(system, seed=0, profile='notebook', retrain=False, training='staged', output=None, live=True, **options):
    if system not in ('sphere', 'o3'):
        raise ValueError('Choose sphere or o3.')
    settings = {'pilot': (1024, 256, 256, 1200, 1600),
                'notebook': (4096, 512, 512, 8000, 8000)}[profile]
    config = dict(system=system, seed=seed, width=32, layers=2, batch_size=256,
        learning_rate=1e-3, minimum_learning_rate=2e-5, cycle_weight=.1,
        spread_weight=.01, grid_depth=7, train_starts=settings[0],
        validation_starts=settings[1], test_starts=settings[2],
        autoencoder_steps=settings[3], latent_steps=settings[4], horizon=10,
        rollout_steps=[1, 4, 10], map_warmup_steps=1200,
        training_boundary_fraction=1/16, validation_boundary_fraction=1/16)
    if training == 'joint':
        config.update(training='joint', parameterization='direct', trajectory_steps=1, data_seed=0,joint_steps=8000 if profile=='notebook' else 1200,
            autoencoder_steps=0, latent_steps=0, map_warmup_steps=0,
            rollout_steps=[1], consistency_weight=1., time_step=1. if system=='sphere' else float(__import__('math').log(3)),
            cmgdb_iterations=10 if system=='sphere' else 1)
    elif training != 'staged':
        raise ValueError('Choose staged or joint training.')
    if set(options)-set(config):
        raise ValueError(f'Unknown settings: {set(options)-set(config)}')
    config.update(options)
    if not 0 <= config['training_boundary_fraction'] <= .5:
        raise ValueError('training_boundary_fraction must lie in [0, 0.5].')
    identity = dict(config=config, implementation=digest(__file__),
        runtime=digest(ROOT/'src/quotient_runtime.py'),
        simulator=digest(ROOT/'vendor/latent-equivariant/code/examples/systems.py'),
        requirements=digest(ROOT/'requirements-emlp.txt'))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    parent = ROOT/'build/quotient-fits'/system
    parent.mkdir(parents=True, exist_ok=True)
    pointer = parent/f'{key}.json'
    if pointer.exists() and not retrain:
        run = ROOT/json.loads(pointer.read_text())['run']
        result = json.loads((run/'result.json').read_text())
        if digest(run/'models.npz') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed; use retrain=True.')
        print(f'Loaded raw-state {system} quotient fit (seed {seed}).')
        if live:
            from training_progress import show_loss
            show_loss(run)
        return run
    run = Path(output).resolve()/'fit' if output is not None else Path(tempfile.mkdtemp(prefix=key+'-', dir=parent))
    if output is not None:run.mkdir(parents=True,exist_ok=False)
    (run/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (run/'provenance.json').write_text(json.dumps(identity, indent=2)+'\n')
    (run/'implementation.py').write_text(Path(__file__).read_text())
    (run/'runtime.py').write_text((ROOT/'src/quotient_runtime.py').read_text())
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT/'build/matplotlib'),
               PYTHONDONTWRITEBYTECODE='1', JAX_ENABLE_X64='true')
    print(f'Fitting raw-state {system} encoder, orbit decoder and quotient map.', flush=True)
    from training_progress import run_with_progress
    returncode=run_with_progress([str(ROOT/'.venv-emlp/bin/python'),'-B',__file__,str(run)],run,env,live=live)
    if returncode:
        raise RuntimeError(f"Fit failed: {run/'training.log'}\n"+(run/'training.log').read_text()[-2500:])
    json.loads((run/'result.json').read_text())
    pointer.write_text(json.dumps(dict(run=str(run.relative_to(ROOT))), indent=2)+'\n')
    print('Saved checkpoint, orbit errors and sampled CMGDB inputs.')
    return run


def _worker(run):
    import random
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import objax
    from emlp.groups import O
    from emlp.reps import T, V
    from emlp.nn.objax import EMLP, Standardize
    from importlib.metadata import version
    from quotient_runtime import make_models, load_quotient
    sys.path.insert(0, str(ROOT/'vendor/latent-equivariant/code/examples'))
    import systems
    cfg = json.loads((run/'config.json').read_text())
    name, seed = cfg['system'], cfg['seed']
    sphere = name == 'sphere'
    dim, qdim = (2, 1) if sphere else (3, 2)
    group, raw_rep, qrep = O(dim), T(0)+V, qdim*T(0)
    input_scale = 1. if sphere else 1.5
    started = time.perf_counter()

    def construct():
        return make_models(cfg)

    models = construct()
    encoder, decoder, flow = models
    # Initialize the quotient step at identity using the library's final layer.
    flow.network[-1].w.assign(jnp.zeros_like(flow.network[-1].w.value))
    flow.network[-1].b.assign(jnp.zeros_like(flow.network[-1].b.value))

    def constrain(q):
        return jnp.tanh(q) if sphere else jnp.concatenate((q[:, :1], jax.nn.softplus(q[:, 1:])), axis=1)
    def unconstrain(q):
        if sphere:
            return jnp.arctanh(jnp.clip(q, -1+1e-12, 1-1e-12))
        return jnp.concatenate((q[:, :1], q[:, 1:]+jnp.log(-jnp.expm1(-jnp.maximum(q[:, 1:], 1e-12)))), axis=1)
    def pack(x):
        return x[:, [2, 0, 1]] if sphere else x
    chart_center, chart_scale = jnp.zeros(qdim), jnp.ones(qdim)
    joint=cfg.get('training')=='joint'
    gain=cfg.get('parameterization')=='gain'
    if gain and sphere:raise ValueError('Gain coordinates are implemented only for O(3).')
    dynamic_chart=joint and sphere
    def chart():
        if dynamic_chart:
            poles=jnp.array([[0.,0.,-1.],[0.,0.,1.]])
            ends=constrain(encoder(pack(poles)/raw_scale))[:,0]
            center=(ends[0]+ends[1])/2; scale=(ends[1]-ends[0])/2
            scale=jnp.where(jnp.abs(scale)>1e-5,scale,jnp.where(scale>=0,1e-5,-1e-5))
            return jnp.array([center]),jnp.array([scale])
        return chart_center,chart_scale
    def E(x):
        if gain:
            base=jnp.column_stack((x[:,0],jnp.linalg.norm(x[:,1:],axis=1)))
            return base*jnp.exp(.25*jnp.tanh(encoder(pack(x)/raw_scale)))
        center,scale=chart()
        return (constrain(encoder(pack(x)/raw_scale))-center)/scale
    def D(q):
        if gain:return q*jnp.exp(.5*jnp.tanh(decoder(q)))
        center,scale=chart()
        return constrain(decoder(q*scale+center))
    def section(q):
        if sphere:
            return jnp.concatenate((jnp.sqrt(jnp.maximum(0, 1-q*q)), jnp.zeros_like(q), q), axis=1)
        return jnp.concatenate((q, jnp.zeros((len(q), 2))), axis=1)
    def invariant(x):
        return x[:, 2:3] if sphere else jnp.column_stack((x[:, 0], jnp.linalg.norm(x[:, 1:], axis=1)))
    def orbit_squared(q, x):
        target = invariant(x)
        if sphere:
            return jnp.sum((q-target)**2+(jnp.sqrt(jnp.maximum(1e-15, 1-q*q))-
                                         jnp.sqrt(jnp.maximum(0, 1-target*target)))**2, axis=1)
        return jnp.sum((q-target)**2, axis=1)

    splits = {}
    for label, offset in [('train', 1000), ('validation', 2000), ('test', 10000)]:
        count = cfg[label+'_starts']
        x = systems.sample_states(name, count, seed=cfg.get('data_seed',seed)+offset)
        if label != 'test':
            # Include the physical boundary/axis in training without providing E targets.
            k = int(count*cfg.get('training_boundary_fraction',1/16)) if label == 'train' else int(count*cfg.get('validation_boundary_fraction',1/16))
            if sphere:
                x[:k] = [0., 0., -1.]; x[k:2*k] = [0., 0., 1.]
            else:
                x[:k, 1:] = 0.; x[k:2*k, 0] = 0.
        y = systems.step(name, x, duration=cfg.get('time_step',systems.SPACING[name]),corotating=sphere)
        splits[label] = (jnp.asarray(x), jnp.asarray(y))
    tx, ty = splits['train']; vx, vy = splits['validation']; testx, testy = splits['test']
    np.savez_compressed(run/'training_pairs.npz',
        **{label+'_'+suffix:np.asarray(pair[i]) for label,pair in splits.items()
           for i,suffix in enumerate(('x','y'))})
    data = jnp.concatenate((tx, ty)); validation = jnp.concatenate((vx, vy))
    packed = np.asarray(pack(data))
    raw_scale = jnp.asarray([np.sqrt(np.mean(packed[:, 0]**2))] +
                           [np.sqrt(np.mean(packed[:, 1:]**2))]*dim)
    rng = np.random.default_rng(seed+7300)
    history = []
    def optimize(stage, loss, variables, arrays, checks, steps):
        gradient = objax.GradValues(loss, variables)
        optimizer = objax.optimizer.Adam(variables)
        def update(lr, *batch):
            gradients, values = gradient(*batch); optimizer(lr, gradients)
            return values[0]
        update = objax.Jit(update, gradient.vars()+optimizer.vars())
        evaluate = objax.Jit(loss, models.vars())
        best = initial = float(evaluate(*checks))
        path = str(run/(stage+'.npz'))
        objax.io.save_var_collection(path, variables)
        for k in range(steps):
            ids = rng.integers(len(arrays[0]), size=cfg['batch_size'])
            lr = cfg['minimum_learning_rate']+.5*(cfg['learning_rate']-cfg['minimum_learning_rate'])*(1+np.cos(np.pi*k/max(steps-1, 1)))
            value = float(update(lr, *[a[ids] for a in arrays]))
            if not np.isfinite(value):
                raise FloatingPointError(f'{stage}: nonfinite loss at {k+1}')
            if (k+1)%200 == 0 or k+1 == steps:
                current = float(evaluate(*checks))
                row = dict(stage=stage, update=k+1, train=value, validation=current)
                history.append(row); print(json.dumps(row), flush=True)
                if np.isfinite(current) and current < best:
                    best = current; objax.io.save_var_collection(path, variables)
        objax.io.load_var_collection(path, variables)
        return dict(initial_validation_loss=initial, best_validation_loss=best, steps=steps)
    def ae_loss(x):
        q = E(x); rec = D(q)
        cycle = jnp.mean((E(section(rec))-q)**2)
        spread = jnp.mean((jnp.std(q, axis=0)-.5)**2)
        return jnp.mean(orbit_squared(rec, x))/input_scale**2 + cfg['cycle_weight']*cycle + cfg['spread_weight']*spread
    ae_vars = objax.ModuleList([encoder, decoder]).vars()
    ae = None if joint else optimize('orbit_autoencoder', ae_loss, ae_vars, [data], [validation], cfg['autoencoder_steps'])
    frozen = [np.array(v.value) for v in ae_vars.values()]
    if sphere and not joint:
        ends = E(jnp.array([[0., 0., -1.], [0., 0., 1.]]))[:, 0]
        chart_center = jnp.array([(ends[0]+ends[1])/2])
        chart_scale = jnp.array([(ends[1]-ends[0])/2])
        if abs(float(chart_scale[0])) < 1e-3:
            raise ValueError('The sphere encoder collapsed the poles.')
    if joint:
        # Fixed normalization during joint optimization; latent targets move with E.
        latent_center,latent_scale=jnp.zeros(qdim),jnp.ones(qdim)
    else:
        encode, decode = objax.Jit(E, encoder.vars()), objax.Jit(D, decoder.vars())
        z, znext = encode(tx), encode(ty); zv, zvnext = encode(vx), encode(vy)
        latent_center, latent_scale = jnp.mean(z, axis=0), jnp.maximum(jnp.std(z, axis=0), 1e-4)
    network = Standardize(flow, (latent_center, latent_scale, jnp.zeros(qdim), jnp.ones(qdim)))
    def F(q):
        increment = network(q, training=False)
        if gain:return q*jnp.exp(3*jnp.tanh(increment/3))
        if sphere:
            # Boundary-preserving interval output head; no analytical flow supplied.
            shift = jnp.tanh(3*jnp.tanh(increment/3))
            return (q+shift)/(1+q*shift)
        return constrain(unconstrain(q)+increment)
    def flow_loss(q, qnext):
        return jnp.mean(((F(q)-qnext)/latent_scale)**2)
    def rollout_loss(target):
        prediction, loss = target[:, 0], 0.
        for step in range(1, max(cfg['rollout_steps'])+1):
            prediction = F(prediction)
            if step in cfg['rollout_steps']:
                loss += jnp.mean(((prediction-target[:, step])/latent_scale)**2)
        return loss/len(cfg['rollout_steps'])
    joint_changes=None
    if joint:
        initial=[[np.array(v.value) for v in m.vars().values()] for m in models]
        def joint_loss(x,y):
            q=E(x);pred=F(q)
            return (ae_loss(jnp.concatenate((x,y)))+
                jnp.mean(orbit_squared(D(pred),y))/input_scale**2+
                cfg['consistency_weight']*jnp.mean(((pred-E(y))/latent_scale)**2))
        warmup=None
        joint_arrays=[tx,ty];joint_checks=[vx,vy]
        if cfg.get('trajectory_steps',1)>1:
            # Keep time-one targets, adding later trajectory pairs from the same
            # independent starts. No axis anchors or repeated special points.
            pairs={}
            for label,x in [('train',tx),('validation',vx)]:
                states=[np.asarray(x)]
                for _ in range(cfg['trajectory_steps']):
                    states.append(systems.step(name,states[-1],duration=cfg['time_step'],corotating=sphere))
                trajectory=np.stack(states,axis=1)
                pairs[label]=[jnp.asarray(trajectory[:,:-1].reshape(-1,dim+1)),
                              jnp.asarray(trajectory[:,1:].reshape(-1,dim+1))]
            joint_arrays=pairs['train'];joint_checks=pairs['validation']
            np.savez_compressed(run/'trajectory_pairs.npz',**{label+'_'+suffix:np.asarray(pair[i])
                for label,pair in pairs.items() for i,suffix in enumerate(('x','y'))})
        dynamics=optimize('joint',joint_loss,models.vars(),joint_arrays,joint_checks,cfg['joint_steps'])
        joint_changes={label:float(np.sqrt(sum(np.sum((a-np.asarray(v.value))**2)
            for a,v in zip(before,m.vars().values()))))
            for label,before,m in zip(('encoder','decoder','latent_map'),initial,models)}
        assert all(v>0 for v in joint_changes.values()),joint_changes
        if sphere:chart_center,chart_scale=chart()
        dynamic_chart=False
        encode,decode=objax.Jit(E,encoder.vars()),objax.Jit(D,decoder.vars())
    else:
        warmup=optimize('map_warmup',flow_loss,flow.vars(),[z,znext],[zv,zvnext],cfg['map_warmup_steps'])
        target_train=systems.trajectories(name,np.asarray(tx),intervals=max(cfg['rollout_steps']),corotating=sphere)
        target_val=systems.trajectories(name,np.asarray(vx),intervals=max(cfg['rollout_steps']),corotating=sphere)
        target_train=encode(target_train.reshape(-1,dim+1)).reshape(len(tx),-1,qdim)
        target_val=encode(target_val.reshape(-1,dim+1)).reshape(len(vx),-1,qdim)
        dynamics=optimize('quotient_map',rollout_loss,flow.vars(),[target_train],[target_val],cfg['latent_steps'])
        assert all(np.array_equal(old,np.asarray(v.value)) for old,v in zip(frozen,ae_vars.values()))
    advance = objax.Jit(F, flow.vars())
    objax.io.save_var_collection(str(run/'models.npz'), models.vars())
    def batched(fn, x):
        return np.concatenate([np.asarray(fn(jnp.asarray(x[i:i+2048]))) for i in range(0, len(x), 2048)])
    def error(q, x):
        return float(jnp.sqrt(jnp.mean(orbit_squared(jnp.asarray(q), jnp.asarray(x))))/input_scale)
    test_q = batched(encode, np.asarray(testx))
    rec = batched(decode, test_q)
    prediction = batched(decode, batched(advance, test_q))
    metrics = dict(reconstruction_orbit_rms=error(rec, testx), prediction_orbit_rms=error(prediction, testy))
    metrics['closure_scaled_rms'] = float(np.sqrt(np.mean(((batched(advance, test_q)-batched(encode, np.asarray(testy)))/np.asarray(latent_scale))**2)))
    state, latent = np.asarray(testx), test_q.copy()
    for _ in range(cfg['horizon']):
        state = systems.step(name, state,duration=cfg.get('time_step',systems.SPACING[name]),corotating=sphere)
        latent = batched(advance, latent)
    metrics['rollout_orbit_rms'] = error(batched(decode, latent), state) if np.isfinite(latent).all() else None
    metrics['rollout_horizon'] = cfg['horizon']
    # Orthogonal invariance, checked on raw states without an observation wrapper.
    rotation = np.eye(dim); angle = .731
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    reflection = np.eye(dim); reflection[0, 0] = -1
    invariance = {}
    for label, matrix in [('rotation', rotation), ('reflection', reflection)]:
        changed = np.asarray(testx).copy()
        if sphere: changed[:, :2] = changed[:, :2]@matrix.T
        else: changed[:, 1:] = changed[:, 1:]@matrix.T
        invariance[label] = float(np.max(np.abs(batched(encode, changed)-test_q)))
    assert max(invariance.values()) < 2e-6, invariance
    # Plot and grid geometry. E receives raw states; F acts directly on q.
    n = 2**cfg['grid_depth']
    if sphere:
        physical = np.linspace(-1, 1, 401)[:, None]
        raw = np.column_stack((np.sqrt(1-physical[:, 0]**2), np.zeros(len(physical)), physical[:, 0]))
        qplot = np.linspace(-1, 1, 401)[:, None]
        lower, upper = np.array([-1.]), np.array([1.])
        grid = np.linspace(-1, 1, n+1)
        boxes = np.column_stack((grid[:-1], grid[1:]))
    else:
        a, r = np.meshgrid(np.linspace(-1.5, 1.5, 19), np.linspace(0, 1.5, 13))
        physical = np.column_stack((a.ravel(), r.ravel()))
        raw = np.column_stack((physical, np.zeros((len(physical), 2))))
        allq = batched(encode, np.asarray(data)); span = np.ptp(allq, axis=0)
        lower = np.array([allq[:, 0].min()-.1*span[0], 0.])
        upper = allq.max(axis=0)+.1*span
        a, r = np.meshgrid(np.linspace(lower[0], upper[0], 15), np.linspace(0, upper[1], 12))
        qplot = np.column_stack((a.ravel(), r.ravel()))
        a, r = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        lo = lower+np.column_stack((a.ravel(), r.ravel()))*(upper-lower)/n
        boxes = np.column_stack((lo, lo+(upper-lower)/n))
    encoded = batched(encode, raw)
    if sphere:
        increments = np.diff(encoded[:, 0])/np.diff(physical[:, 0])
        geometry = dict(sampled_monotone=bool(np.all(increments>0)),
                        derivative_range=[float(increments.min()),float(increments.max())],
                        sampled_encoder_range=[float(encoded.min()),float(encoded.max())])
        if not geometry['sampled_monotone'] or encoded.min() < -1-1e-10 or encoded.max() > 1+1e-10:
            raise ValueError('Sphere encoder does not pass the sampled interval-coordinate check.')
    else:
        a, r = np.meshgrid(np.linspace(-1.49,1.49,81),np.linspace(.01,1.49,61))
        checkq = jnp.array(np.column_stack((a.ravel(),r.ravel())))
        def quotient_encode(q):
            return E(jnp.concatenate((q,jnp.zeros(2)))[None,:])[0]
        jac = np.asarray(jax.jit(jax.vmap(jax.jacfwd(quotient_encode)))(checkq))
        determinants=np.linalg.det(jac)
        geometry=dict(sampled_jacobian_points=len(checkq),
                      determinant_range=[float(determinants.min()),float(determinants.max())],
                      determinant_changes_sign=bool(determinants.min()<0<determinants.max()),
                      minimum_singular_value=float(np.linalg.svd(jac,compute_uv=False)[:,-1].min()))
    iterations = cfg.get('cmgdb_iterations',10 if sphere else 1)
    def iterate(q):
        for _ in range(iterations): q = batched(advance, q)
        return q
    rawnext = systems.step(name, raw, duration=iterations*cfg.get('time_step',systems.SPACING[name]), corotating=sphere)
    fractions = np.linspace(0, 1, 5)[:, None] if sphere else np.array([(a, b) for a in np.linspace(0, 1, 5) for b in np.linspace(0, 1, 5)])
    sample = boxes[:, None, :qdim]+fractions[None, :, :]*(boxes[:, None, qdim:]-boxes[:, None, :qdim])
    im = iterate(sample.reshape(-1, qdim)).reshape(len(boxes), len(fractions), qdim)
    assert np.isfinite(im).all()
    image_boxes = np.column_stack((im.min(axis=1)-1e-9, im.max(axis=1)+1e-9))
    if sphere: image_boxes = np.clip(image_boxes, -1, 1)
    else: image_boxes[:, 1] = np.maximum(image_boxes[:, 1], 0)
    np.savez_compressed(run/'evaluation.npz', physical_quotient=physical, encoded_quotient=encoded,
        encoded_next=batched(encode, rawnext), predicted_encoded_next=iterate(encoded),
        quotient_grid=qplot, quotient_images=iterate(qplot), raw_states=np.asarray(testx),
        decoded_quotient=rec, reconstructed_states=np.asarray(section(jnp.asarray(rec))),
        boxes=boxes, image_boxes=image_boxes, domain_lower=lower, domain_upper=upper,
        raw_scale=np.asarray(raw_scale), chart_center=np.asarray(chart_center), chart_scale=np.asarray(chart_scale),
        latent_center=np.asarray(latent_center), latent_scale=np.asarray(latent_scale))
    _, restored_E, restored_D, restored_F = load_quotient(run)
    reload_error = max(float(np.max(np.abs(a-b))) for a,b in [
        (batched(restored_E,raw),encoded), (batched(restored_D,test_q),rec),
        (batched(restored_F,test_q),batched(advance,test_q))])
    assert reload_error < 1e-12, reload_error
    result = dict(config=cfg, group=f'O({dim})', encoder_input='raw state',
        raw_dimension=dim+1, quotient_dimension=qdim, representation='T(0)+V -> '+str(qdim)+'*T(0)',
        architecture='Official emlp.nn.objax.EMLP; all-scalar quotient map and orbit decoder',
        synthetic_observations=False, analytic_encoder_targets=False,
        explicit_invariant_skip=gain,
        invariant_prior='(a, norm(v)) with positive learned gains' if gain else None,
        normalization=dict(raw='RMS scalar; one shared RMS vector-component scale',
                           latent='identity during joint training' if joint else 'frozen mean and standard deviation'),
        metrics=metrics, invariance_max_errors=invariance, autoencoder=ae, map_warmup=warmup, dynamics=dynamics,
        checkpoint_sha256=digest(run/'models.npz'), frozen_encoder_decoder_unchanged=not joint,
        training_mode='joint' if joint else 'staged',joint_parameter_changes=joint_changes,
        checkpoint_reload_max_error=reload_error, encoder_geometry=geometry,
        quotient_iterations=iterations, certified=False,
        cmgdb_input='5 samples per coordinate per cell; hull plus 1e-9; NOT validated',
        split_seeds={k: cfg.get('data_seed',seed)+i for k, i in [('train',1000),('validation',2000),('test',10000)]},
        versions={p: version(p) for p in ('emlp','objax','jax','jaxlib','numpy','scipy')},
        elapsed_seconds=time.perf_counter()-started)
    (run/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    (run/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    _worker(Path(sys.argv[1]).resolve())
