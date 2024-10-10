import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import glob
import json
import shutil
import argparse

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

def create_gif(directory:str, savepath:str, filename:str, duration:float=0.02, loop=0):
    images = [imageio.v2.imread(filename) for filename in sorted(glob.glob(os.path.join(directory, '*.png')))]
    imageio.mimsave(os.path.join(savepath, f'{filename}.gif'), images, duration=duration, loop=loop)
    

def graph_array(y, title:str, xlabel:str, ylabel:str, label:str, savepath:str=None):
    plt.plot(y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    if savepath:
        plt.savefig(f'{savepath}/{title}.png')
    else:
        plt.show()
    plt.close()


def plot_loss(losses:list, labels:list[str], *, savepath: str = None, filename:str=None, clip_begin:bool = False):
    plt.figure(figsize=(10, 6))
    for loss, label in zip(losses, labels):
        if clip_begin:
            loss = loss[5:]
        plt.plot(loss, label=label)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss over Epochs', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    if savepath:
        plt.savefig(f'{savepath}/{filename}.png', bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def create_dense_network(num_layers:int, num_neurons:int, input_size:int, output_size:int, activation:tworch.layer.Activation):
    net = [
        tworch.layer.DenseLayer(input_size, num_neurons),
        activation(),
        *[layer for _ in range(num_layers - 1) for layer in (tworch.layer.DenseLayer(num_neurons, num_neurons), activation())],
        tworch.layer.DenseLayer(num_neurons, output_size)
    ]
    return net



INPUT_SIZE = 1
OUTPUT_SIZE = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network to fit a dataset')
    parser.add_argument('--layers', type=int, help='Number of layers in the network')
    parser.add_argument('--neurons', type=int, help='Number of neurons in each layer')
    parser.add_argument('--activation', type=str, help='Activation function to use')
    parser.add_argument('--dataset_name', type=str, help='Path to the dataset')
    parser.add_argument('--save', type=bool, help='Save the results', default=False)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.015)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=100)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=20)

    args = parser.parse_args()


    # remove the plots folder and recreate it
    plots_path = os.path.join(this_file_path, 'plots')
    if os.path.exists(plots_path):
        shutil.rmtree(plots_path)
    os.makedirs(plots_path)
    reports_path = os.path.join(this_file_path, f'reports/{args.activation}')
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    json_path = os.path.join(reports_path, 'json')
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    loss_plot_path = os.path.join(reports_path, 'loss_plots')
    if not os.path.exists(loss_plot_path):
        os.makedirs(loss_plot_path)

    
    dataset_path = os.path.join(this_file_path, 'datasets')
    dataset_name = args.dataset_name
    dataset = np.load(f'{dataset_path}/{dataset_name}(500, 2).npy')

    x = dataset[:, 0]
    y = dataset[:, 1]
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    n = len(x[0])

    # Network parameters
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    train_losses = []

    layers = args.layers
    neurons = args.neurons
    if args.activation == 'ReLU':
        activation = tworch.layer.ReLU
    elif args.activation == 'Tanh':
        activation = tworch.layer.Tanh
    elif args.activation == 'Sigmoid':
        activation = tworch.layer.Sigmoid
    elif args.activation == 'Softplus':
        activation = tworch.layer.Softplus
    
    # Create the network
    net = create_dense_network(num_layers=layers, num_neurons=neurons, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, activation=activation)
    print(f'Created a network with {layers} layers, {neurons} neoruns, and {args.activation} activation')
    # Create the fitting line input
    xmin, xmax = np.min(x), np.max(x)
    x_linspace = np.linspace(xmin, xmax, 100).reshape(1, -1) # 1x100
    # setup for results
    configuration_name = f'{layers} layer {neurons} nn-{args.activation} ({dataset_name})'

    # Train the network
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        for i in range(0, n, batch_size):
            input = x[:, i:i+batch_size]
            target = y[:, i:i+batch_size]
            forward_output = forward(input, net)
            train_loss += np.sum((forward_output - target) ** 2)
            delta = 2 * (forward_output - target) 
            backward(delta, net, learning_rate)

        train_losses.append(train_loss/n)
        save_fitting_line(net, x_linspace, dataset, title=f'Fitting line epoch:{epoch+1:0>4} with {configuration_name.split("(")[0][:-1]}, loss: {train_loss / n:.4f}', savepath=f'{this_file_path}/plots')
    print(f'Training done for {epochs} epochs')
    
    create_gif(directory=f'{plots_path}', savepath=f'{reports_path}', filename=f'fitting {configuration_name}', duration=0.1)
    print(f'Gif created')

    plot_loss([train_losses], ['train loss'], savepath=f'{loss_plot_path}', filename=f'loss {configuration_name}', clip_begin=True)
    print(f'Loss graph created')

    # save the losses as json
    with open(f'{json_path}/losses {configuration_name}.json', 'w') as f:
        json.dump({'train': train_losses}, f)
    print(f'Losses saved as json')


