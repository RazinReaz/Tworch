# Feed-Forward Neural Network
## Model class
To use a feed-forward neural network, you can create a new model by calling the `FNN` class.
``` python
model = tworch.network.FNN(input_size, output_size, learning_rate, batch_size, epochs)
```
## Methods
### Sequential
``` python
model.sequential(layer_1, layer_2, ...) -> None
```
This method takes a list of layers and adds them to the model in the order they are given. The order of the layers matter.  An example block of layers may be as follows:
`DenseLayer -> Activation -> Dropout -> ...` 
### Forward

``` python
model.forward(x, training = True) -> None
```
`x` is the input to the model. This method takes the input `x` and passes it through the model architecture.
`training` is a boolean value that is used to determine if dropout should be applied or not.

### Backward
``` python
model.backward(target) -> None
```
`target` is the target output of the model. This method calculates the loss and the gradients of the model.

### Train
``` python
model.train(X, y) -> None
```
Trains the model on input data `X` and target labels `y`.
### Predict

``` python
model.predict(x) -> int
```
This method gives the result of the model for a single instance  `x`.
### Accuracy

``` python
model.accuracy(predictions, target) -> float
```
Calculates accuracy given predictions and target labels

### Export
``` python
model.export(filepath : str) -> None
```
This method saves the weights and architecture of a trained model as a `pickle` file to the `filepath`.
### Describe
``` python
model.describe() -> None
```
This method prints the architecture of the model.
### Create Model
``` python
load_model(filepath : str) -> FNN
```
This method creates a trained model from an exported `pickle` file.
