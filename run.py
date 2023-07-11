#!python

import inspect
import json
import modal
import os
import random
import argparse
from uuid import uuid4 as uuid

from game import game
from eval import train as _train

stub = modal.Stub(name="blockeval")
image = modal.Image.debian_slim().pip_install(
    "crayons",
    "recordtype",
    "jax[cpu]",
    "jax",
)

@stub.function(image=image)
def random_game(_):
    return game.random_game()

@stub.function(image=image)
def train(data):
    return _train(data)

@stub.local_entrypoint()
def main(games_path: str):
    n_games = 10000

    # Load games
    print(f'Loading games from {games_path}...')
    games = []
    for game_file in os.listdir(games_path)[:n_games]:
        print(f'Loading {game_file}')
        with open(os.path.join(games_path, game_file), 'r') as f:
            games.append(game.load_game(json.load(f)))
    print('Loaded')

    # Generate games
    if (len(games) < n_games):
        print(f'Generating {n_games - len(games)} random games...')
        new_games = random_game.map(range(n_games - len(games)))
        for g in new_games:
            with open(os.path.join(games_path, str(uuid()) + '.game'), 'w') as f:
                f.write(g.json)
        games += new_games
    

    print('We have the games')
    
    # Train a model on the games
    data = [(g.masks[-1], g.winners) for g in games]
    print('Training model...')
    evaluator = train.call(data)

if __name__ == '__main__':
    sig = inspect.signature(main.raw_f)
    parser = argparse.ArgumentParser()
    for param in sig.parameters.values():
        print(param.name)
        parser.add_argument(f'--{param.name}', required=True)
    args = parser.parse_args()
    main(**vars(args))