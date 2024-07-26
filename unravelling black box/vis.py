
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import glob

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
this_file_path = os.path.dirname(os.path.abspath(__file__))
import tworch

def forward(x, net):
    for layer in net:
        x = layer(x)
    return x

def backward(delta_output, net, learning_rate):
    for layer in reversed(net):
        delta_output = layer.backward(delta_output, learning_rate)
    return delta_output

def save_fitting_line(net, x, dataset, title, savepath:str):
    model = net.copy()
    output = forward(x, model)

    # plt.xlim(-1, 1)
    # plt.ylim(-0.5, 1.5)
    plt.plot(x[0], output[0], color='red', label='fitting line')
    plt.scatter(dataset[:, 0], dataset[:, 1], s=1, label='dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f'{savepath}/fitting_line_at_{epoch:0>4}.png')
    plt.close()

def create_gif(directory:str):
    # create a gif from the images of the synthetic data
    images = []
    for filename in glob.glob(f'{this_file_path}/{directory}/*.png'):
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{this_file_path}/fitting.gif', images, duration=0.02)

if __name__ == "__main__":
    dataset = np.load(f'{this_file_path}/dataset(500, 2).npy')

    x = dataset[:, 0]
    y = dataset[:, 1]
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    print(f'{x.shape=}, {y.shape=}')

    net = []
    net.append(tworch.layer.DenseLayer(1, 4))
    net.append(tworch.layer.Sigmoid())
    net.append(tworch.layer.DenseLayer(4, 1))

    learning_rate = 0.01
    epochs = 400
    losses = []
    input = x
    target = y
    xmin, xmax = np.min(x), np.max(x)
    x_linspace = np.linspace(xmin, xmax, 100).reshape(1, -1) # 1x100

    for epoch in tqdm(range(epochs)):
        forward_output = forward(input, net)
        # print(f'{forward_output.shape=}, {target.shape=}')
        delta = (forward_output - target)**2
        loss = np.mean(delta)
        losses.append(loss)
        backward(delta, net, learning_rate)
        save_fitting_line(net, x_linspace, dataset, title=f'Fitting line at epoch {epoch:0>4}, loss: {loss}', savepath=f'{this_file_path}/plots')

    print(f'Training done for {epochs} epochs')
    create_gif(f'plots')
    print(f'Gif created')

    # xmin, xmax = np.min(x), np.max(x)
    # x_linspace = np.linspace(xmin, xmax, 100).reshape(1, -1) # 1x100
    # first_layer_output = net[0].weights @ x_linspace + net[0].bias # 3x100
    # print(f'{first_layer_output.shape=}')
    # activation_output = net[1](first_layer_output) # 3x100
    # second_layer_output = net[2].weights @ activation_output + net[2].bias # 1x100

    # plt.scatter(x, y, s=1)
    # plt.plot(x_linspace[0], second_layer_output[0], color='red', )
    # plt.show()

