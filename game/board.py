from random import random, shuffle

from recordtype import recordtype

from .ominos import TOTAL_OMINOS, TOTAL_TILES, Transformation, get_omino_positions, get_omino_score

PLAYERS = 4
WIDTH = 20
HEIGHT = 20

Cell = recordtype('Cell', ['i', 'j', 'val'])

def in_bounds(cells, positions):
    for i, j in positions:
        if i < 0 or i > WIDTH or j < 0 or j > HEIGHT:
            return False
    return True


def vacant(cells, positions):
    for i, j in positions:
        if i < 0 or i >= HEIGHT or j < 0 or j >= WIDTH:
            return False
        if cells[(i * WIDTH) + j].val != 0:
            return False
    return True


def shares_edge_with(cells, positions, player):
    for [ci, cj] in positions:
        for [di, dj] in [[1,0],[-1,0],[0,1],[0,-1]]:
            i = ci + di
            j = cj + dj
            if i < 0 or j < 0 or i >= HEIGHT or j >= WIDTH:
                continue
            if cells[(i * WIDTH) + j].val == player:
                return True
    return False


def shares_vertex_with(cells, positions, player):
    for [ci, cj] in positions:
        # Initial positions
        if (ci in [0, WIDTH - 1] and cj in [0, HEIGHT - 1]):
            return True
        for [di, dj] in [[1,1],[-1,1],[1,-1],[-1,-1]]:
            i = ci + di
            j = cj + dj
            if i < 0 or j < 0 or i >= HEIGHT or j >= WIDTH:
                continue
            if cells[(i * WIDTH) + j].val == player:
                return True
    return False


def validate_place(b, player, omino_idx, transformation, x, y):
    positions = get_omino_positions(omino_idx, transformation, x, y)
    if not in_bounds(b.cells, positions):
        return False
    if not vacant(b.cells, positions):
        return False
    if shares_edge_with(b.cells, positions, player):
        return False
    if not shares_vertex_with(b.cells, positions, player):
        return False
    return True


def valid_moves(b, player, randomize = True):
    ominos = list(b.ominos_remaining[player])
    columns = list(range(WIDTH))
    rows = list(range(HEIGHT))
    r = list(range(4))
    f = list(range(2))
    if randomize:
        for l in (ominos, columns, rows, r, f):
            shuffle(l)
    for omino_idx in ominos:
        for x in columns:
            for y in rows:
                for rotations in r:
                    for flips in f:
                        transformation = Transformation(rotations, flips)
                        if validate_place(b, player, omino_idx, transformation, x, y):
                            yield (omino_idx, transformation, x, y)


def has_valid_move(b, player):
    # hack for performance
    for move in valid_moves(b, player, True):
        b._next_move = move
        return move
    # return any(valid_moves(b, player, true))


def random_move(b, player):
    # hack for performance (see corresponding hack in has_valid_move)
    if b._next_move:
        return b._next_move
    for move in valid_moves(b, player, True):
        return move

def uniformly_random_move(b, player):
    for move in valid_moves(b, player, True):
        return move

def get_next_player(b, player):
    for i in range(PLAYERS - 1):
        next_player = (player + i) % PLAYERS + 1
        if has_valid_move(b, next_player):
            return next_player
    return None


def score(b, player):
    score = TOTAL_TILES
    for omino_idx in b.ominos_remaining[player]:
        score -= get_omino_score(omino_idx)
    return score

class Board:
    def __init__(self, cols = WIDTH, rows = HEIGHT, players = PLAYERS):
        self.cells = []
        for n in range(cols * rows):
            self.cells.append(Cell(n // cols, n % cols, 0))

        self.ominos_remaining = {
            1: set(range(1, TOTAL_OMINOS + 1)),
            2: set(range(1, TOTAL_OMINOS + 1)),
            3: set(range(1, TOTAL_OMINOS + 1)),
            4: set(range(1, TOTAL_OMINOS + 1))
        }

        self.next_player = 1
        
        #hack, see has_valid_move and random_move
        self._next_move = None

        self.alive = {
            1: True,
            2: True,
            3: True,
            4: True
        }

        self.game_over = False

    def __str__(self):
        s = ('-' * 20) + '\n'
        for i in range(WIDTH):
            for j in range(HEIGHT):
                s += str(self.cells[(i * WIDTH) + j].val or ' ')
            s += '\n'
        s += '-' * 20
        return s

    def place(self, player, omino_idx, transformation, x, y):
        assert player == self.next_player
        assert validate_place(self, player, omino_idx, transformation, x, y)
        assert omino_idx in self.ominos_remaining[player]
        
        positions = get_omino_positions(omino_idx, transformation, x, y)
        for i, j in positions:
            self.cells[(i * WIDTH) + j].val = player
        self.ominos_remaining[player].remove(omino_idx)
        
        self.next_player = get_next_player(self, player)
        if not self.next_player:
            self.game_over = True

def place_cells(cells, player, omino_idx, transformation, x, y):
    positions = get_omino_positions(omino_idx, transformation, x, y)
    for i, j in positions:
        cells[(i * WIDTH) + j].val = player

def unplace_cells(cells, player, omino_idx, transformation, x, y):
    positions = get_omino_positions(omino_idx, transformation, x, y)
    for i, j in positions:
        cells[(i * WIDTH) + j].val = 0

