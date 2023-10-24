from data import Loader

loader = Loader('/Users/samclearman/games')
[all] = loader.samples([100000])
X, Y = all
print(len([y for y in Y if sum(y) != 1]))