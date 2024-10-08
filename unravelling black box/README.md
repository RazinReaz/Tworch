# Unravelling the black box

## Creating a 2D dataset
- Run `dataset_creation.py` to create a 2D dataset
    - Currently, two kinds of dataset are supported: `down-up-down` and `stairs`.
    - The dataset is saved in the working directory.

## Run the code
- To run the code, simply install the required packages and run `run.sh`.
- You can edit the `run.sh` file to change the dataset, number of layers, activation used, learning rate, etc.

## Visualizations
### Tanh activation (1 layer neural network with 2 neurons)
![Tanh activation](reports/fitting%201%20layer%202%20nn%20-%20Tanh.gif)
### Sigmoid activation (1 layer neural network with 2 neurons)
![Sigmoid activation](reports/fitting%201%20layer%202%20nn%20-%20Sigmoid.gif)
### ReLU activation (1 layer neural network with 10 neurons)
![ReLU activation](reports/fitting%201%20layer%2010%20nn%20-%20ReLU.gif)
### ReLU activation (2 layer neural network with 10 neurons each)
![ReLU activation](reports/fitting%202%20layer%2010%20nn%20-%20ReLU.gif)
