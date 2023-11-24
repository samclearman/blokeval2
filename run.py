from eval import load_params
from game.game import random_game
from inspect_params import prediction_logits_and_loss

g = random_game()

print('Generated game')
print(g)

params = load_params('params.npz')
prediction, logits, l = prediction_logits_and_loss(params, g)
print(f'Prediction: {prediction}  Loss: {l} \n (logits: {debug["logits"]})')