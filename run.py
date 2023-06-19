import json
import modal
import os
import random
from uuid import uuid4 as uuid

from game import game

stub = modal.Stub(name="blockeval")
image = modal.Image.debian_slim().pip_install(
    "crayons",
    "recordtype"
)

@stub.function(image=image)
def random_game(_):
    return game.random_game()

@stub.local_entrypoint()
def main(games_path: str):
    n_games = 10

    # Load games
    print(f'Loading games from {games_path}...')
    games = []
    for game_file in os.listdir(games_path)[:n_games]:
        with open(os.path.join(games_path, game_file), 'r') as f:
            games.append(game.load_game(json.load(f)))

    # Generate games
    if (len(games) < n_games):
        print(f'Generating {n_games - len(games)} random games...')
        new_games = random_game.map(range(n_games - len(games)))
        for g in new_games:
            with open(os.path.join(games_path, str(uuid()) + '.game'), 'w') as f:
                f.write(g.json)
        games += new_games
