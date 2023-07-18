#!python

import inspect
import json
import modal
import os
import random
import argparse
from uuid import uuid4 as uuid

import jax.numpy as jnp

from game import game
from eval import train as _train

stub = modal.Stub(name="blockeval")
basic_image = modal.Image.debian_slim().pip_install(
    "crayons",
    "recordtype",
    "jax[cpu]",
    "jax",
)
training_image = modal.Image.debian_slim(force_build=True).run_commands(
    'pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
).pip_install(
    "crayons",
    "recordtype",
)

@stub.function(image=basic_image)
def random_game(_):
    return game.random_game()

@stub.function(image=training_image, gpu="any")
def train(games, arrays):
    if (len(arrays[0])):
        X, Y = arrays
        def batch():
            for i in range(1000):
                idxs = random.sample(range(int(len(X) * 0.8)), 128)
                yield (X[idxs], Y[idxs])
        training_batches = batch()
        test_set = (X[int(len(X) * 0.8):], Y[int(len(X) * 0.8):])
        return _train(training_batches, test_set)

    data = [(g.masks[-1], g.winners) for g in games]
    def batch():
        for _ in range(1000):
            yield data_to_jnp_arrays(random.sample(data[:int(len(data) * 0.8)], 128))
    training_batches = batch()
    test_set = data_to_jnp_arrays(data[int(len(data) * 0.8):])
    return _train(training_batches, test_set)

def data_to_jnp_arrays(data):
    return jnp.array([r[0] for r in data]), jnp.array([r[1] for r in data])

@stub.local_entrypoint()
def main(games_path: str = None, games_file: str = None):
    n_games = 10000

    X = None; Y = None
    if games_file:
        with jnp.load(games_file, 'r') as f:
            X, Y = (f['X'], f['Y'])
    n_games -= len(X)

    games = []
    if (games_path):
        # Load games
        print(f'Loading games from {games_path}...')
        for game_file in os.listdir(games_path):
            if (not game_file.endswith('.full')):
                continue
            with open(os.path.join(games_path, game_file), 'r') as f:
                games.append(game.load_game(json.load(f)))
                if len(games) >= n_games:
                    break
        print('Loaded')

    # Generate games
    if (len(games) < n_games):
        print(f'Generating {n_games - len(games)} random games...')
        new_games = random_game.map(range(n_games - len(games)))
        for g in new_games:
            with open(os.path.join(games_path, str(uuid()) + '.game'), 'w') as f:
                f.write(g.json)
        games += new_games

    # Train a model on the games
    print('Training model...')
    evaluator = train(games=games, arrays=(X, Y))


if __name__ == '__main__':
    sig = inspect.signature(main.raw_f)
    parser = argparse.ArgumentParser()
    for param in sig.parameters.values():
        print(param.name)
        parser.add_argument(f'--{param.name}', required=False)
    args = parser.parse_args()
    main(**vars(args))
