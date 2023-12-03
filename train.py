#!python

import inspect
import argparse
import time
from uuid import uuid4 as uuid
from itertools import islice

from eval import train as _train, save_params
from cloud import stub, training_image
from data import Loader, batched, exactly_one_winner, shuffle_players

@stub.function(image=training_image, gpu="any")
def train(training_batches, test_set):
    return _train(training_batches, test_set)


@stub.local_entrypoint()
def main(games_path: str = None):
    n_training_games = 800000
    n_test_games = 80000
    batch_size = 512
    n_batches = 100000

    # n_training_games = 800
    # n_test_games = 80
    # batch_size = 10
    # n_batches = 10

    # Train a model on the games
    print('Loading samples...')
    loader = Loader(games_path)
    loader.filter(exactly_one_winner)
    # loader.transform(shuffle_players)
    training_set, test_set = loader.samples([n_training_games, n_test_games])
    print('Training model...')
    start = time.perf_counter()
    evaluator, params = train(islice(batched(training_set, batch_size), n_batches), test_set)
    duration = time.perf_counter() - start
    print(f'trained on {n_batches} batches in {duration} seconds - {duration / n_batches} sec/batch')
    save_params('params.npz', params)


if __name__ == '__main__':
    sig = inspect.signature(main.raw_f)
    parser = argparse.ArgumentParser()
    for param in sig.parameters.values():
        print(param.name)
        parser.add_argument(f'--{param.name}', required=False)
    args = parser.parse_args()
    main(**vars(args))
