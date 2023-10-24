import modal

stub = modal.Stub(name="blockeval")

basic_image = modal.Image.debian_slim().pip_install(
    "crayons",
    "recordtype",
    "jax[cpu]",
    "jax",
)

training_image = modal.Image.debian_slim(force_build=True).run_commands(
    'pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
).pip_install(
    "crayons",
    "recordtype",
)
