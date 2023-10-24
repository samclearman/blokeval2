#!/usr/bin/env python3

import argparse
import json
import os
from itertools import islice
from uuid import uuid4 as uuid

import jax.numpy as jnp

from data import data_to_jnp_arrays
from game import game


parser = argparse.ArgumentParser()
parser.add_argument('games_path')
parser.add_argument('--batch-size', type=int, default=1000000000, required=False)
parser.add_argument('--final-positions-only', action=argparse.BooleanOptionalAction, required=False)
args = parser.parse_args()

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

print(args.games_path)
fullfiles = [f for f in os.listdir(args.games_path) if f.endswith('.full')]

for batch in batched(fullfiles, args.batch_size):
    games = []
    for game_file in batch:
        with open(os.path.join(args.games_path, game_file), 'r') as f:
            games.append(game.load_game(json.load(f)))
    if args.final_positions_only:
        data = [(g.masks[-1], g.winners) for g in games]
    else:
        data = flatten([[(mask, g.winners) for mask in g.masks] for g in games])
    X, Y = data_to_jnp_arrays(data)
    name = os.path.join(args.games_path, 'batch_' + str(uuid()) + '.npz')
    jnp.savez(name, X=X, Y=Y)