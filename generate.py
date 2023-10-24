#!/usr/bin/env python3

import os
from uuid import uuid4 as uuid

import jax.numpy as jnp

from data import generate, data_to_jnp_arrays, STORAGE_BATCH_SIZE
from cloud import stub

# parser = argparse.ArgumentParser()
# parser.add_argument('batch_dir')
# args = parser.parse_args()

@stub.local_entrypoint()
def main(batch_dir):
    games = generate(STORAGE_BATCH_SIZE)

    for g in games:
        with open(os.path.join(batch_dir, str(uuid()) + '.game.full'), 'w') as f:
            f.write(g.to_json(full = True))

    data = [(g.masks[-1], g.winners) for g in games]
    X, Y = data_to_jnp_arrays(data)
    name = os.path.join(batch_dir, 'final_positions' + '.npz')
    jnp.savez(name, X=X, Y=Y)