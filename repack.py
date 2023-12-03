#!/usr/bin/env python3

import argparse
import json
import os
from itertools import islice
from uuid import uuid4 as uuid
import random as pyrandom

import jax.numpy as jnp

from data import data_to_jnp_arrays
from game import game

packfiles = [f for f in os.listdir(os.getcwd()) if f.startswith('batch')]
print(len(packfiles))

Xs = []
Ys = []

for f in packfiles:
    with jnp.load(f, 'r') as combined:
        Xs.append(combined['X'])
        Ys.append(combined['Y'])

X = jnp.concatenate(Xs)
print(X.shape)
Y = jnp.concatenate(Ys)
print(Y.shape)
jnp.savez('batch.npz', X=X, Y=Y)