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
    logits = dense(params, input)
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

def init_params_handpicked(key):
    s = 20 * 20
    w = jnp.array([[1.0,0,0,0]] * s + [[0.0,1,0,0]] * s + [[0.0,0,1,0]] * s + [[0.0,0,0,1]] * s)
    b = jnp.array([0.0, 0, 0, 0])
    return ((w, b),)

n_epochs = 1
lr = 0.1
def train(training_batches, test_set):
    seed = pyrandom.randint(0, 1000)
    print("Seed {}".format(seed))
    key = random.PRNGKey(seed)
    params = init_params(key)
    test_inputs, test_targets = test_set
    print(f'{"":12} {"Train":25} {"Test":25}')
    print(f'{"Epoch":12} {"Accuracy":12} {"Loss":12} {"Accuracy":12} {"Loss":12}')
    for epoch in range(n_epochs):
        for batch in training_batches:
            train_inputs, train_targets = batch
            training_loss = loss(params, train_inputs, train_targets)
            training_accuracy = accuracy(params, train_inputs, train_targets)
            test_loss = loss(params, test_inputs, test_targets)
            test_accuracy = accuracy(params, test_inputs, test_targets)
            print(f'{epoch:12} {training_accuracy:12.4f} {training_loss:12.4f} {test_accuracy:12.4f} {test_loss:12.4f}')
            params = update(params, train_inputs, train_targets, lr)
    def evaluator(input):
        return predict(params, input)
    return evaluator