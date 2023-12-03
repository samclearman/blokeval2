import random as pyrandom

import jax.numpy as jnp
import jax.nn as jnn
import jax
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

def predict(params, input, debug = {}):
    logits = dense(params, input)
    if 'logits' in debug:
        debug['logits'] = logits
    return jnp.argmax(logits)

def error(params, input, target):
    logits = dense(params, input)
    # print('-----logits-----')
    # print(logits)
    # print('-----softmax----')
    # print(jnn.softmax(logits))
    return jnp.dot(-jnn.log_softmax(logits), target)
batched_error = vmap(error, in_axes=(None, 0, 0))

def loss(params, inputs, targets):
    errs = batched_error(params, inputs, targets)
    return jnp.mean(errs)

def acc(params, input, target):
    return (predict(params, input) == jnp.argmax(target)).astype(jnp.float32)
batched_accuracy = vmap(acc, in_axes=(None, 0, 0))

def accuracy(params, inputs, targets):
    accs = batched_accuracy(params, inputs, targets)
    return jnp.mean(accs)

@jit
def update(params, inputs, targets, lr, debug = {}):
    grads = grad(loss)(params, inputs, targets)
    if 'norm' in debug:
        debug['norm'] = jnp.linalg.norm(jnp.concatenate([jnp.ravel(w) for w, _ in grads]))
    if 'norms' in debug:
        debug['norms'] = [(jnp.linalg.norm(w), jnp.linalg.norm(b)) for w, b in grads]
        # for i, (w, b) in enumerate(grads):
        #     print(f'layer {i} gradient norms, w: {jnp.linalg.norm(w)}, b: {jnp.linalg.norm(b)}')
        #     print(w.shape)
        #     print(w)
        #     print(b.shape)
        #     print(b)
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]

def _init_params(key):
    widths = [1600, 800, 400, 200, 4]
    # widths = [20 * 20 * 4, 4]
    for input_width, output_width, k in zip(widths[:-1], widths[1:], random.split(key, len(widths) - 1)):
        kw, kb = random.split(k)
        w = random.normal(kw, (input_width, output_width)) * M1
        b = random.normal(kb, (output_width,)) * M2
        yield w, b
def init_params(key):
    return tuple(_init_params(key))

def init_params_handpicked(key):
    s = 20 * 20
    w = jnp.array([[1.0,0,0,0]] * s + [[0.0,1,0,0]] * s + [[0.0,0,1,0]] * s + [[0.0,0,0,1]] * s)
    b = jnp.array([0.0, 0, 0, 0])
    return ((w, b),)

def save_params(path, params):
    jnp.savez(path, *[a for t in params for a in t])

def load_params(path):
    with jnp.load(path, 'r') as d:
        keys = iter(d.keys())
        params = [(d[k1], d[k2]) for k1, k2 in zip(keys, keys)]
        return params

def make_evaluator(params):
    def evaluator(input, debug = {}):
        return predict(params, input, debug)
    return evaluator

# jax.config.update("jax_transfer_guard", "log")
n_epochs = 1
lr = 0.1
M1 = 0.01
M2 = 0
def train(training_batches, test_set):
    seed = pyrandom.randint(0, 1000)
    print("Seed {}".format(seed))
    print(f'Magic numbers: {lr=} {M1=} {M2=}')
    key = random.PRNGKey(seed)
    params = init_params(key)
    test_inputs, test_targets = test_set
    print(f'{"":12} {"Train":25} {"Test":25}')
    print(f'{"Batch":12} {"Accuracy":12} {"Loss":12} {"Accuracy":12} {"Loss":12}            |Δ|')
    for i, batch in enumerate(training_batches):
        train_inputs, train_targets = batch
        training_loss = loss(params, train_inputs, train_targets)
        training_accuracy = accuracy(params, train_inputs, train_targets)
        test_loss = loss(params, test_inputs, test_targets)
        test_accuracy = accuracy(params, test_inputs, test_targets)
        # debug = { 'norm': None, 'norms': None }
        debug = { 'norm': None }
        params = update(params, train_inputs, train_targets, lr, debug)
        print(f'{i:12} {training_accuracy:12.4f} {training_loss:12.4f} {test_accuracy:12.4f} {test_loss:12.4f}            {debug["norm"]}')
        # for w, b in debug['norms']:
        #     print(f'{w:6}, {b:6}')
    return (make_evaluator(params), params)