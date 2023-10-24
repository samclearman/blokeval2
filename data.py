import json
import os
import random as pyrandom

import jax.numpy as jnp

from cloud import stub, basic_image
from game import game

STORAGE_BATCH_SIZE = 10000

@stub.function(image=basic_image)
def random_game(_):
    return game.random_game()

def data_to_jnp_arrays(data):
    return jnp.array([r[0] for r in data]), jnp.array([r[1] for r in data])

def load_batch(batch_dir):
    X = None; Y = None
    if (os.path.exists(os.path.join(batch_dir, 'final_positions.npz'))):
        with jnp.load(os.path.join(batch_dir, 'final_positions.npz'), 'r') as combined:
            return combined['X'], combined['Y']

    # Load games
    games = []
    print(f'Loading games from {batch_dir}...')
    for game_file in [filename for filename in os.listdir(batch_dir) if filename.endswith('.full')]:
        with open(os.path.join(batch_dir, game_file), 'r') as f:
            games.append(game.load_game(json.load(f)))
    return data_to_jnp_arrays(games)

def generate(n_games):
    return random_game.map(range(n_games))

class Loader:
    def __init__(self, games_dir):
        self.games_dir = games_dir
        self.filters = []
    
    @property
    def batch_dirs(self):
        return [os.path.join(self.games_dir, d) for d in os.listdir(self.games_dir) if os.path.isdir(os.path.join(self.games_dir, d))]

    @property
    def dataset_size(self):
        return STORAGE_BATCH_SIZE * len(self.batch_dirs)
    
    def filter(self, f):
        self.filters.append(f)

    def samples(self, ns):
        if sum(ns) > self.dataset_size:
            raise ValueError(f'Cannot sample {sum(ns)} games from {self.dataset_size}')
        samples = []

        offset = 0
        for n in ns:
            loaded = []
            while n > 0 and offset < self.dataset_size:
                start = offset % STORAGE_BATCH_SIZE
                incr = min(n, 10000 - start)
                X, Y = load_batch(self.batch_dirs[offset // STORAGE_BATCH_SIZE])
                mask = jnp.ones(len(X), dtype=bool)
                for f in self.filters:
                    mask &= f(X, Y)
                X = X[mask]
                Y = Y[mask]
                loaded.append((X[start:start + incr], Y[start:start + incr]))
                offset += incr
                n -= len(X)
            samples.append((jnp.concatenate([X for X, _ in loaded]), jnp.concatenate([Y for _, Y in loaded])))
            if n > 0:
                raise ValueError(f'Not enough data in dataset to sample {ns} games')
        return samples

def batched(arrays, batch_size):
    X, Y = arrays
    print(X.shape)
    print(Y.shape)
    def batch():
        while True:
            idxs = jnp.array(pyrandom.sample(range(len(X)), batch_size))
            yield (X[idxs], Y[idxs])
    return batch()

def exactly_one_winner(X, Y):
    return jnp.sum(Y, axis=1) == 1