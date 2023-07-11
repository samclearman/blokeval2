#!/usr/bin/env python3

import argparse
import json

from game import game


parser = argparse.ArgumentParser()
parser.add_argument('game_file')
args = parser.parse_args()

with open(args.game_file) as f:
    g = game.load_game(json.load(f))
    print(g.to_json(full = True))

