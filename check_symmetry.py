from eval import load_params
from inspect_params import support

params = load_params('params.npz')
for i in range(4):
    s = support(params, 3, i)
    t = (s[:400].sum().item(), s[400:800].sum().item(), s[800:1200].sum().item(), s[1200:].sum().item())
    print(f'{i=}: {t}')