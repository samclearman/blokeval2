#!/usr/bin/env python3

import argparse
import json

import jax.numpy as jnp

from run import data_to_jnp_arrays
from game import game


parser = argparse.ArgumentParser()
parser.add_argument('game_file')
args = parser.parse_args()

with open(args.game_file) as f:
    g = game.load_game(json.load(f))
    data = [(mask, g.winners) for mask in g.masks]
    X, Y = data_to_jnp_arrays(data)
    jnp.savez(args.game_file + '.npz', X=X, Y=Y)


