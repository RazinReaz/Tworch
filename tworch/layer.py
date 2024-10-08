import numpy as np
from typing import Callable
from .initializer import *
from .optimizer import *

class Layer():
    def __init__(self):
        pass
    def __call__(self, input):
        pass
    def backward(self, delta_output, learning_rate):
        pass
    def __str__(self) -> str:
        return "Layer"

class DenseLayer(Layer):
    def __init__(self, input_size:int, output_size:int, *, initializer:Initializer = Xavier()):
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.weights = self.initializer(input_size, output_size)
        self.bias = self.initializer(1, output_size)
        self.weight_optimizer = Adam(input_size, output_size)
        self.bias_optimizer = Adam(1, output_size)
        self.name = "Dense"
        
    def __call__(self, input):
        """
        input: (input_size, batch_size)
        output: (output_size, batch_size)
        """
        assert input.shape[0] == self.input_size, f"Dense layer expects input size of {self.input_size} but got {input.shape[0]}"
        self.input = input
        self.output = np.dot(self.weights, input) + self.bias
        return self.output
    
    def __str__(self) -> str:
        return "Dense Layer: (" +str(self.input_size) + "," + str(self.output_size) + ")\n" + str(self.initializer) 
    
    def backward(self, delta_output, learning_rate):
        """
        delta_output: (output_size, batch_size)
        """
        assert delta_output.shape == self.output.shape, f"back propagation expects delta_output of shape {self.output.shape} but got {delta_output.shape}"
        batch_size = delta_output.shape[1]
        self.delta_weights = np.dot(delta_output, self.input.T) / batch_size
        self.delta_bias = np.sum(delta_output, axis=1, keepdims=True) / batch_size

        weights_copy = self.weights.copy()
        self.weights = self.weight_optimizer.update(self.weights, self.delta_weights, learning_rate)
        self.bias = self.bias_optimizer.update(self.bias, self.delta_bias, learning_rate)
        # self.weights -= learning_rate * self.delta_weights
        # self.bias -= learning_rate * self.delta_bias

        return np.dot(weights_copy.T, delta_output)
    
class Dropout(Layer):
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.name = "Dropout"
    def set_size(self, size):
        self.input_size = size
        self.output_size = size
    def __str__(self) -> str:
        return "Dropout: " + str(self.keep_prob)
    def __call__(self, input):
        self.input = input
        self.mask = np.random.binomial(1, self.keep_prob, size=input.shape)
        self.output = (input * self.mask) / self.keep_prob
        return self.output
    def backward(self, delta_output, learning_rate):
        return (delta_output * self.mask) / self.keep_prob

def softmax(x:np.ndarray):
    """
    x: (classes, batch_size)
    """
    x_max = np.max(x, axis=0, keepdims=True)
    x = np.exp(x - x_max)
    probabilities = x / np.sum(x, axis=0, keepdims=True)
    return probabilities


class Activation(Layer):
    def __init__(self, activation: Callable, derivative: Callable, name: str):
        self.activation = activation
        self.derivative = derivative
        self.name = name
    def set_size(self, size):
        self.input_size = size
        self.output_size = size
    def __str__(self) -> str:
        return "Activation: "+ self.name
    def __call__(self, input:np.ndarray):
        assert len(input.shape) == 2, f'Activation layer expects input of shape (input_size, batch_size) but got {input.shape}'
        self.input = input
        self.output = self.activation(input)
        return self.output    
    def backward(self, delta_output:np.ndarray, learning_rate:float):
        return delta_output * self.derivative(self.input)



class Softmax(Activation):
    def __init__(self):
        super().__init__(activation=self.softmax, derivative=self.softmax_prime, name="Softmax")

    def softmax(self, x: np.ndarray):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def softmax_prime(self, x: np.ndarray):
        return self.softmax(x) * (1 - self.softmax(x))

class ReLU(Activation):
    def __init__(self):
        super().__init__(activation=self.relu, derivative=self.relu_prime, name="ReLU")

    def relu(self, x: np.ndarray):
        return np.maximum(x, 0)

    def relu_prime(self, x: np.ndarray):
        return np.where(x > 0, 1, np.where(x < 0, 0, 0.5))

class Tanh(Activation):
    def __init__(self):
        super().__init__(activation=self.tanh, derivative=self.tanh_prime, name="Tanh")

    def tanh(self, x: np.ndarray):
        return np.tanh(x)

    def tanh_prime(self, x: np.ndarray):
        return 1 - np.tanh(x)**2

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(activation=self.sigmoid, derivative=self.sigmoid_prime, name="Sigmoid")

    def sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x: np.ndarray):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

class Linear(Activation):
    def __init__(self):
        super().__init__(activation=self.linear, derivative=self.linear_prime, name="Linear")

    def linear(self, x: np.ndarray):
        return x

    def linear_prime(self, x: np.ndarray):
        return np.ones_like(x)
    
class Softplus(Activation):
    def __init__(self):
        super().__init__(activation=self.softplus, derivative=self.softplus_prime, name="Softplus")

    def softplus(self, x: np.ndarray):
        return np.log(1 + np.exp(x))

    def softplus_prime(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))