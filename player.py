import jax.numpy as jnp

from game.game import flat_mask
from game.board import Cell, place_cells, unplace_cells, valid_moves

def jax_evaluator(model, player):
    def f(board, moves):
        cells = [Cell(*c) for c in board.cells]
        masks = []
        for move in moves:
            place_cells(cells, move)
            masks.append(flat_mask(cells))
            unplace_cells(cells, move)
        X = jnp.array(masks)
        predictions = model.predict(X)
        return predictions[:,player - 1]
    return f

# Right now this is a basic player which just picks whichever move has the best evaluator score
class JaxPlayer:
    def __init__(self, player_idx, model):
        self._evaluator = jax_evaluator(model, player_idx)
        self._player_idx = player_idx

    def next_move(self, game):
        b = game.board
        if b.next_player != self._player_idx:
            raise 'Not my turn'
        moves = list(valid_moves(b, self._player_idx))
        scores = self._evaluator(b, moves)

        best_score = 0
        chosen_move = None

        for move, score in zip(moves, scores):
            if score >= best_score:
                chosen_move = move
                best_score = score

        return chosen_move
