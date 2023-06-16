from game.game import random_game

g = random_game()

print('Game over!')
print(g)
print('History')
for blob in g.blobs:
    print(blob)
