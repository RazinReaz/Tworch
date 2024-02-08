# Tworch
Tworch is a python library for creating and training neural networks.
Currently it is in development and not ready for use.

# How to import
pip install is not supported. Therefore, you can install the package by cloning the repository.

To use `tworch`, open a new python file in the cloned directory and write
``` python
import os
import sys
sys.path.append('path/to/tworch')
import tworch
```
Please Refer to the [training code](training/train.py) or [testing code](evaluate/test.py) for a complete example.

# Documentation
## Feed-Forward Neural Network
### Model class
To use a feed-forward neural network, you can create a new model by calling the `FNN` class.
``` python
model = tworch.network.FNN(input_size, output_size, learning_rate, batch_size, epochs)
```
### Methods
``` python
model.sequential(layer_1, layer_2, ...) -> None
```
This method takes a list of layers and adds them to the model in the order they are given. The order of the layers matter.  An example block of layers may be as follows:
`DenseLayer -> Activation -> Dropout -> ...` 
\
\
\

``` python
model.forward(x, training = True) -> None
```
`x` is the input to the model. This method takes the input `x` and passes it through the model architecture.
`training` is a boolean value that is used to determine if dropout should be applied or not.
\
\
\

``` python
model.backward(target) -> None
```
`target` is the target output of the model. This method calculates the loss and the gradients of the model.
\
\
\

``` python
model.train(X, y) -> None
```
Trains the model on input data `X` and target labels `y`.
\
\
\

``` python
model.predict(x) -> int
```
This method gives the result of the model for a single instance  `x`.
\
\
\

``` python
model.accuracy(predictions, target) -> float
```
Calculates accuracy given predictions and target labels
\
\
\
``` python
model.export(filepath : str) -> None
```
This method saves the weights and architecture of a trained model as a `pickle` file to the `filepath`.
\
\
\
``` python
create_model(filepath : str) -> FNN
```
This method creates a trained model from an exported `pickle` file.
\
\
\
``` python
model.describe() -> None
```
This method prints the architecture of the model.




\
\
\
\
\
\

## Supported Features
- [x] Feed Forward Neural Network
  - [x] Dense layer
  - [x] Dropout
  - [x] Initializers
    - [x] Xavier
    - [x] He
    - [x] LeCum
  - [x] Activation
    - [x] Sigmoid
    - [x] Tanh
    - [x] ReLU
    - [ ] Leaky ReLU
    - [x] Softmax
  - [x] Optimizers
    - [ ] Momentum
    - [ ] RMS Prop
    - [x] Adam
- [ ] Convolutional Neural Network
