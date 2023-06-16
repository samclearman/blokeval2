import json
from crayons import red, green, blue, yellow

from .board import Board, random_move, score
from .ominos import Transformation

PLAYERS = 4
WIDTH = 20
HEIGHT = 20

def cell_values(b):
    return [c.val for c in b.cells]

def flat_mask(cells):
    return sum(masks([c.val for c in cells]), [])
    
def masks(snapshot):
    m = []
    for p in range(1, PLAYERS + 1):
        m.append([1 if i == p else 0 for i in snapshot])
    return m

reprs = (' ', str(red('█')), str(green('█')), str(blue('█')), str(yellow('█')))
def blob(snapshot):
    blob = ''
    for i in range(20):
        blob += ''.join([(reprs[p] * 2) for p in snapshot[20 * i : 20 * (i + 1)]])
        blob += '\n'
    blob = blob[:-1]
    return blob

def random_game():
    g = Game()
    while not g.game_over:
        b = g.board
        player = b.next_player
        omino_idx, transformation, x, y = random_move(b, player)
        g.play_move((player, omino_idx, transformation, x, y))
    return g

def play_game(p1, p2, p3, p4):
    players = (None, p1, p2, p3, p4)
    g = Game()
    while not g.game_over:
        b = g.board
        player = players[b.next_player]
        move = player.next_move(g)
        g.play_move(move)
    return g

def load_game(json):
    g = Game(moves = json['moves'])
    return g

class Game:
    def __init__(self, moves = []):
        self._snapshots = []
        self._scores = {1: 0, 2: 0, 3: 0, 4: 0}
        self._b = Board()
        self._moves = []

        for m in moves:
            self.play_move(m)

    def __str__(self):
        status = (
            str(len(self._moves)) +
            ('<{}> '.format(self._b.next_player) if  not self.game_over else '<☠> ') +
            ':'.join([str(self._scores[p]) for p in self._scores])
        )
        return status + '\n' + str(self._b)

    def play_move(self, move):
        self._b.place(*move)
        self._moves.append(move)
        self._snapshots.append(cell_values(self._b))
        for p in self._scores:
            self._scores[p] = score(self._b, p)

    @property
    def json(self):
        return json.dumps({
            'moves': self._moves,
        })

    @property
    def masks(self):
        return [m1 + m2 + m3 + m4 for (m1, m2, m3, m4) in [masks(s) for s in self._snapshots]]

    @property
    def blobs(self):
        return [blob(s) for s in self._snapshots]
    
    @property
    def board(self):
        # Todo: return a copy of b
        return self._b

    @property
    def scores(self):
        return [self._scores[p] for p in [1,2,3,4]]

    @property
    def winners(self):
        m = max(self.scores)
        return [int(s == m) for s in self.scores]

    @property
    def turns(self):
        return len(self._moves);

    @property
    def game_over(self):
        return self._b.game_over

