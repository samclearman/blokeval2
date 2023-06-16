from collections import namedtuple
from functools import lru_cache
from math import ceil

Position = namedtuple('Position', ['i', 'j'])
Transformation = namedtuple('Transformation', ['rotations', 'flips'])

OMINO_SIZE = 5

OMINOS = [
  [[1]],

  [[1, 1]],

  [[1, 1, 1]],

  [[1, 1],
   [1]],

  [[1, 1, 1, 1]],

  [[1, 1, 1],
   [1]],

  [[1, 1, 1],
   [0, 1]],

  [[1, 1, 0],
   [0, 1, 1]],

  [[1, 1],
   [1, 1]],

  [[1, 1, 1, 1, 1]],

  [[1, 1, 1, 1],
   [1]],

  [[1, 1, 1, 1],
   [0, 1]],

  [[1, 1, 1, 0],
   [0, 0, 1, 1]],

  [[1, 1, 1],
   [1, 1]],

  [[1, 1, 1],
   [1, 0, 1]],

  [[1, 1, 1],
   [1, 0, 0],
   [1, 0, 0]],

  [[1, 1, 1],
   [0, 1, 0],
   [0, 1, 0]],

  [[1, 1, 0],
   [0, 1, 1],
   [0, 0, 1]],

  [[1, 1, 0],
   [0, 1, 1],
   [0, 1, 0]],

  [[0, 1, 0],
   [1, 1, 1],
   [0, 1, 0]],

  [[1, 1, 0],
   [0, 1, 0],
   [0, 1, 1]],

]

TOTAL_OMINOS = len(OMINOS)
TOTAL_TILES = sum(sum(sum(OMINOS, []), []))

def get_omino_score(idx):
    score = 0
    for r in get_omino(idx):
        score += sum(r)
    return score

def get_omino(idx):
    return OMINOS[idx - 1]

def padded(omino):
    height = len(omino)
    N = OMINO_SIZE
    width = max([len(r) for r in omino])
    pad_left = (N - width) // 2
    pad_top = (N - height) // 2
    pad_bottom = ceil((N - height) / 2)


    padded = []
    for i in range(pad_top):
        padded.append([0] * N)

    for i in range(height):
        r = omino[i]
        pad_right = N - (pad_left + len(r))
        padded_row = ([0] * pad_left) + r + ([0] * pad_right)
        padded.append(padded_row)

    for i in range(pad_bottom):
        padded.append([0] * N)

    return padded
                      

def rotated(padded_omino):
    N = len(padded_omino)
    rotated = []
    for i in range(N):
        rotated.append([0] * N)
    for i in range(N):
        for j in range(N):
            rotated[i][j] = padded_omino[N - j - 1][i];
    return rotated


def flipped(padded_omino):
    N = len(padded_omino)
    flipped = []
    for r in padded_omino:
        flipped.append(list(reversed(r)))
    return flipped


def transformed(omino, transformation):
    omino = padded(omino)
    for i in range(transformation.rotations):
        omino = rotated(omino)
    if transformation.flips:
        omino = flipped(omino)
    return omino

@lru_cache(None)
def get_omino_positions(omino_idx, transformation, x, y):
    omino = get_omino(omino_idx)
    t = transformed(omino, transformation)
    positions = []
    offset = OMINO_SIZE // 2
    for i in range(len(t)):
        for j in range(len(t[i])):
            if t[i][j]:
                positions.append(Position(x + i - offset, y + j - offset))
    return positions
