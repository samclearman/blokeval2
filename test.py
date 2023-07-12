# from .game files
import json
import time
from eval import data_to_jnp_arrays
from game.game import load_game

gamefiles = [
    "0000dcc7-bd51-4430-8c55-c1961d7411de.game",
    "000273f8-f897-4b69-9456-0131a696eefb.game",
    "00048b75-a667-4efe-bee8-3d245eafda7f.game",
    "0017d274-f55e-4d6a-a2bc-42928ddcf1f5.game",
    "001be289-8503-48e2-8f4e-29c9fb491cf6.game",
    "001eb27b-bb2e-4564-afb1-c8e551bede13.game",
    "00224d32-8782-4520-9b3a-74fa614ad704.game",
    "00242bd5-7f81-4ec1-bf2b-fbea749e1866.game",
    "00245181-f720-447f-9433-a1d9439d2eaa.game",
    "002a88db-1803-4bb8-871b-1fbe381d7531.game"
]
start = time.perf_counter()
games = []
for file in gamefiles:
    with open(file, 'r') as f:
        game = load_game(json.load(f))
        games.append(game)
data = [(g.masks[-1], g.winners) for g in games]
arrays = data_to_jnp_arrays(data)
print(f'Time to load games: {time.perf_counter() - start:0.2f} seconds')

fullfiles = [
    "0000dcc7-bd51-4430-8c55-c1961d7411de.game.full",
    "000273f8-f897-4b69-9456-0131a696eefb.game.full",
    "00048b75-a667-4efe-bee8-3d245eafda7f.game.full",
    "0017d274-f55e-4d6a-a2bc-42928ddcf1f5.game.full",
    "001be289-8503-48e2-8f4e-29c9fb491cf6.game.full",
    "001eb27b-bb2e-4564-afb1-c8e551bede13.game.full",
    "00224d32-8782-4520-9b3a-74fa614ad704.game.full",
    "00242bd5-7f81-4ec1-bf2b-fbea749e1866.game.full",
    "00245181-f720-447f-9433-a1d9439d2eaa.game.full",
    "002a88db-1803-4bb8-871b-1fbe381d7531.game.full"
]
start = time.perf_counter()
games = []
for file in fullfiles:
    with open(file, 'r') as f:
        game = load_game(json.load(f))
        games.append(game)
data = [(g.masks[-1], g.winners) for g in games]
arrays = data_to_jnp_arrays(data)
print(f'Time to load games: {time.perf_counter() - start:0.2f} seconds')

