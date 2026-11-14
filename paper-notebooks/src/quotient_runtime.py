"""Reload the direct-quotient EMLP models in the pinned Python 3.11 environment."""
import json
import random
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import objax
from emlp.groups import O
from emlp.reps import T, V
from emlp.nn.objax import EMLP, Standardize


def make_models(cfg):
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); objax.random.DEFAULT_GENERATOR.seed(cfg['seed'])
    dimension, qdim = (2, 1) if cfg['system']=='sphere' else (3, 2)
    group, qrep = O(dimension), qdim*T(0)
    return objax.ModuleList([
        EMLP(T(0)+V, qrep, group=group, ch=cfg['width'], num_layers=cfg['layers']),
        EMLP(qrep, qrep, group=group, ch=cfg['width']*T(0), num_layers=cfg['layers']),
        EMLP(qrep, qrep, group=group, ch=cfg['width']*T(0), num_layers=cfg['layers'])])


def load_quotient(run):
    run=Path(run)
    cfg=json.loads((run/'config.json').read_text())
    models=make_models(cfg)
    objax.io.load_var_collection(str(run/'models.npz'), models.vars())
    with np.load(run/'evaluation.npz') as data:
        scale, center, spread, lc, ls = [jnp.asarray(data[k]) for k in
            ('raw_scale','chart_center','chart_scale','latent_center','latent_scale')]
    sphere=cfg['system']=='sphere'
    gain=cfg.get('parameterization')=='gain'
    def constrain(q):
        return jnp.tanh(q) if sphere else jnp.concatenate((q[:,:1],jax.nn.softplus(q[:,1:])),axis=1)
    def encode(x):
        packed=x[:,[2,0,1]] if sphere else x
        if gain:
            base=jnp.column_stack((x[:,0],jnp.linalg.norm(x[:,1:],axis=1)))
            return base*jnp.exp(.25*jnp.tanh(models[0](packed/scale)))
        return (constrain(models[0](packed/scale))-center)/spread
    def decode(q):
        if gain:return q*jnp.exp(.5*jnp.tanh(models[1](q)))
        return constrain(models[1](q*spread+center))
    network=Standardize(models[2],(lc,ls,jnp.zeros_like(lc),jnp.ones_like(ls)))
    def advance(q):
        delta=network(q,training=False)
        if gain:return q*jnp.exp(3*jnp.tanh(delta/3))
        if sphere:
            shift=jnp.tanh(3*jnp.tanh(delta/3))
            return (q+shift)/(1+q*shift)
        inverse=jnp.concatenate((q[:,:1],q[:,1:]+jnp.log(-jnp.expm1(-jnp.maximum(q[:,1:],1e-12)))),axis=1)
        return constrain(inverse+delta)
    return cfg, objax.Jit(encode,models[0].vars()), objax.Jit(decode,models[1].vars()), objax.Jit(advance,models[2].vars())
