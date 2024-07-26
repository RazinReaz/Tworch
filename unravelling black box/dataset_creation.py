import numpy as np
import matplotlib.pyplot as plt
import os

def f(x):
    return np.sin(x * 2 * np.pi) / 2 + 0.5

num_points = 500
xmin, xmax = -1, 1
xrange = xmax - xmin

dataset = []

for i in range(num_points):
    x = np.random.uniform(xmin, xmax)
    if (x - xmin) / xrange < 0.2:
        y = 0 + np.abs(np.random.normal(0, 0.01))
    elif (x - xmin) / xrange < 0.8:
        y = 1 - np.abs(np.random.normal(0, 0.01))
    else:
        y = 0 + np.abs(np.random.normal(0, 0.01))
    dataset.append([x, y])
dataset = np.array(dataset)
print(f'{dataset.shape=}')

plt.scatter(*zip(*dataset), s=1)
plt.show()

this_file_path = os.path.dirname(os.path.abspath(__file__))
np.save(f'{this_file_path}/dataset{dataset.shape}.npy', dataset)
print(f'dataset saved')
