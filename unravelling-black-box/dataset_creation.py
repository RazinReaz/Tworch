import numpy as np
import matplotlib.pyplot as plt
import os

def f(x):
    return np.sin(x * 2 * np.pi) / 2 + 0.5

def down_up_down(*, n:int, xmin:float, xmax:float):
    xrange = xmax - xmin
    dataset = []
    for i in range(n):
        x = np.random.uniform(xmin, xmax)
        noise = np.abs(np.random.normal(0, 0.01))
        if (x - xmin) / xrange < 0.2:
            y = 0 + noise
        elif (x - xmin) / xrange < 0.8:
            y = 1 - noise
        else:
            y = 0 + noise
        dataset.append([x, y])
    return np.array(dataset)

def stairs(*, n:int, xmin:float, xmax:float):
    xrange = xmax - xmin
    dataset = []
    for i in range(n):
        x = np.random.uniform(xmin, xmax)
        ratio = (x - xmin) / xrange
        noise = np.abs(np.random.normal(0, 0.01))
        if ratio < 0.2:
            y = 0 + noise
        elif ratio < 0.4:
            y = 0.25 + noise
        elif ratio < 0.6:
            y = 0.5 + noise
        elif ratio < 0.8:
            y = 0.75 + noise
        else:
            y = 1 - noise
        dataset.append([x, y])
    return np.array(dataset)

def gaussian(*, n:int, xmin:float, xmax:float):
    xrange = xmax - xmin
    dataset = []
    mean = (xmin + xmax) / 2
    std = 0.05
    for i in range(n):
        x = np.random.uniform(xmin, xmax)
        noise = np.abs(np.random.normal(0, 0.05))
        y =  np.exp(-(0.5 * (x - mean)/std)** 2) + noise
        dataset.append([x, y])
    return np.array(dataset)

num_points = 500
dataset_name = f"gaussian"

if dataset_name == "down-up-down":
    dataset = down_up_down(n=num_points, xmin=0, xmax=1)
elif dataset_name == "stairs":
    dataset = stairs(n=num_points, xmin=0, xmax=1)
elif dataset_name == "gaussian":
    dataset = gaussian(n=num_points, xmin=0, xmax=1)
else:
    raise ValueError(f"Unknown dataset name: {dataset_name}")


print(f'{dataset.shape=}')

this_file_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(this_file_path, f'datasets')
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
plt.scatter(*zip(*dataset), s=1)
plt.savefig(f'{dataset_path}/{dataset_name}{dataset.shape}.png')
np.save(f'{dataset_path}/{dataset_name}{dataset.shape}.npy', dataset)
print(f'dataset saved in {dataset_path}')
