import random as pyrandom

import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap, random

def relu(x):
    return jnp.maximum(0, x)

def dense(params, input):
    output = None
    for w, b in params:
        output = jnp.dot(input, w) + b
        input = relu(output)
    return output
batched_dense = vmap(dense, in_axes=(None, 0))

def probs(params, input):
    return jnn.softmax(dense(params, input))

def predict(params, input):
    return jnp.argmax(dense(params, input))


def error(params, input, target):
    raw = dense(params, input)
    return jnp.dot(-jnn.log_softmax(raw), target)
batched_error = vmap(error, in_axes=(None, 0, 0))

def loss(params, inputs, targets):
    errs = batched_error(params, inputs, targets)
    return jnp.mean(errs)

# @jit
def update(params, inputs, targets, lr):
    grads = grad(loss)(params, inputs, targets)
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]

def _init_params(key):
    widths = [20 * 20 * 4, 4]
    for input_width, output_width, k in zip(widths[:-1], widths[1:], random.split(key, len(widths) - 1)):
        kw, kb = random.split(k)
        w = random.normal(kw, (input_width, output_width))
        b = random.normal(kb, (output_width,))
        yield w, b
def init_params(key):
    return tuple(_init_params(key))

def data_to_jnp_arrays(data):
    return jnp.array([r[0] for r in data]), jnp.array([r[1] for r in data])

n_epochs = 1000
lr = 0.001
def train(data):
    # split data into inputs and targets
    train_inputs, train_targets = data_to_jnp_arrays(data[:int(len(data) * 0.8)])
    test_inputs, test_targets = data_to_jnp_arrays(data[int(len(data) * 0.8):])

    seed = pyrandom.randint(0, 1000)
    print("Seed {}".format(seed))
    key = random.PRNGKey(seed)
    params = init_params(key)
    for epoch in range(n_epochs):
        training_loss = loss(params, train_inputs, train_targets)
        test_loss = loss(params, test_inputs, test_targets)
        print(f'Epoch {epoch} train loss: {training_loss}, test loss: {test_loss}')
        params = update(params, train_inputs, train_targets, lr)
    def evaluator(input):
        return predict(params, input)
    return evaluator