# Unravelling the black box

## Creating a 2D dataset
- Run `dataset_creation.py` to create a 2D dataset
    - Supported datasets are
        - `down-up-down`
        - `stairs`
        - `gaussian`
    - The dataset is saved in the `datasets` directory along with the 2-Dimensional plot of the dataset.

## Run the code
- To run the code, simply install the required packages and run `run.sh`.
- You can edit the `run.sh` file to change the dataset, number of layers, activation used, learning rate, etc.
### Arguments
`--layers` : Number of dense layers in the neural network \
`--neurons` : Number of neurons in each layer \
`--activation` : Activation function to be used in the neural network. The activation will be applied to each layer of the neural network. \
`--epochs` : Number of epochs to train the neural network. Default is 100 \
`--lr` : Learning rate of the neural network. Default is 0.015 \
`--batch_size` : Batch size of the neural network. Default is 20 \
`--dataset_name` : Dataset to be used for training the neural network. Check [this](#creating-a-2d-dataset) section for supported datasets. \


## About the model
- The model is a feed forward neural network.
- Adam optimizer is used at every dense layer.
- the `--neurons` argument is used to specify the number of neurons in each layer. So if you specify `--neurons 10`, the model will have 10 neurons in all layers.

## Visualizations
Firstly, the results of a 5 layered neural network with 10 neurons in each layer is shown. The dataset is *gaussian*. We present the learning procedure of the model with different activation functions. The model is trained for 150 epochs. The learning rate is 0.015 and the batch size is 20. 
![comparison of activation functions](reports/Compiled/gaussian%205layer-10nn.gif)\
The gif shows that ReLU activation function is learning consistently and achieves the lowest loss rating fast.
Other visualizations are shown below:
### Tanh activation on **down-up-down** (2 layer neural network with 10 neurons)
![Tanh activation](reports/Tanh/fitting%202%20layer%2010%20nn-Tanh%20(down-up-down).gif)
### Sigmoid activation on **down-up-down** (2 layer neural network with 10 neurons)
![Sigmoid activation](reports/Sigmoid/fitting%202%20layer%2010%20nn-Sigmoid%20(down-up-down).gif)
### ReLU activation on **gaussian** (1 layer neural network with 10 neurons)
![ReLU activation](reports/ReLU/fitting%201%20layer%2010%20nn-ReLU%20(gaussian).gif)
### ReLU activation (5 layer neural network with 10 neurons each)
![ReLU activation](reports/ReLU/fitting%205%20layer%2010%20nn-ReLU%20(gaussian).gif)
