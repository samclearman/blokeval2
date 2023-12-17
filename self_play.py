#!python

import inspect
import argparse
import random as pyrandom
from itertools import islice
from game.game import play_game

from model import Model, init_params as _init_params, init_rng_and_params, update
from cloud import stub, training_image
from data import data_to_jnp_arrays, transformed, shuffle_players
from player import JaxPlayer



lr = 0.1
@stub.local_entrypoint()
def main():
    n_games = 3
    seed = pyrandom.randint(0, 1000)
    print("Seed {}".format(seed))
    key, params = init_rng_and_params(seed)
    for _ in range(n_games):
        g = play_game(*[JaxPlayer(i, Model(params)) for i in range(1,5)])
        tuples = [(mask, g.winners) for mask in g.masks]
        X, Y = data_to_jnp_arrays(tuples)
        X, Y = transformed(X,Y,[shuffle_players])
        params = update(params, X, Y, lr)


if __name__ == '__main__':
    sig = inspect.signature(main.raw_f)
    parser = argparse.ArgumentParser()
    for param in sig.parameters.values():
        print(param.name)
        parser.add_argument(f'--{param.name}', required=False)
    args = parser.parse_args()
    main(**vars(args))
