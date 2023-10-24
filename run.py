#!python

import inspect
import argparse
from uuid import uuid4 as uuid
from itertools import islice

from eval import train as _train
from cloud import stub, training_image
from data import Loader, batched, exactly_one_winner

@stub.function(image=training_image, gpu="any")
def train(training_batches, test_set):
    return _train(training_batches, test_set)


@stub.local_entrypoint()
def main(games_path: str = None):
    n_training_games = 80000
    n_test_games = 8000
    batch_size = 1024
    n_batches = 10000

    # Train a model on the games
    print('Training model...')
    loader = Loader(games_path)
    loader.filter(exactly_one_winner)
    training_set, test_set = loader.samples([n_training_games, n_test_games])
    evaluator = train(islice(batched(training_set, batch_size), n_batches), test_set)


if __name__ == '__main__':
    sig = inspect.signature(main.raw_f)
    parser = argparse.ArgumentParser()
    for param in sig.parameters.values():
        print(param.name)
        parser.add_argument(f'--{param.name}', required=False)
    args = parser.parse_args()
    main(**vars(args))
