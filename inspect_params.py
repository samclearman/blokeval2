import jax.numpy as jnp

from model import make_evaluator, loss
from data import data_to_jnp_arrays

def max_outputs(params):
    max_vals = [jnp.array([1] * 1600)]
    for w, b in params:
        m = []
        for ww, bb in zip(w.T, b):
            m.append(max((jnp.maximum(ww * max_vals[-1], 0).sum() + bb).item(), 0))
        max_vals.append(jnp.array(m))
    return max_vals

def support(params, i, j):
    s = jnp.zeros_like(params[i][0][0]).at[j].set(1)
    for w, b in reversed(params[:i+1]):
        s = jnp.maximum(w @ s, 0)
    return s

def prediction_logits_and_loss(params, game):
    evaluator = make_evaluator(params)
    data = [(game.masks[-1], game.winners)]
    Xs, Ys = data_to_jnp_arrays(data)
    debug = { 'logits': None }
    prediction = evaluator(Xs, debug)
    l = loss(params, Xs, Ys)
    return prediction, debug['logits'], l

