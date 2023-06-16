import modal

from game import game

stub = modal.Stub(name="blockeval")
image = modal.Image.debian_slim().pip_install(
    "crayons",
    "recordtype"
)

@stub.function(image=image)
def random_game(_):
    return game.random_game()

@stub.local_entrypoint()
def main():
    # Genrate training data
    print('Generating random games...')
    games = random_game.map(range(10))
    for game in games:
        print(game.blobs[-1])
        print("")