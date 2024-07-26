import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm

from .layer import *
from .loss import *
from .utils import one_hot

class FNN():
    def __init__(self, input_size:int, output_size:int, *, learning_rate:float=0.0005, batch_size:int=50, epochs:int=100, loss:Loss = CrossEntropyLoss()):
        self.input_size = input_size
        self.output_size = output_size
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.children = []

        self.built = False

        self.loss = loss
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
    
    def __add_layer(self, layer:Layer)->None:
        if len(self.children) == 0:
            assert isinstance(layer, DenseLayer)
            assert layer.input_size == self.input_size
            self.children.append(layer)
            return
        if isinstance(layer, DenseLayer):
            assert layer.input_size == self.children[-1].output_size
        elif isinstance(layer, Activation):
            assert isinstance(self.children[-1], DenseLayer)
            layer.set_size(self.children[-1].output_size)
        elif isinstance(layer, Dropout):
            assert isinstance(self.children[-1], Activation)
            layer.set_size(self.children[-1].output_size)
        self.children.append(layer)
    
    def sequential(self, *layers:Layer)->None:
        for layer in layers:
            self.__add_layer(layer)
        self.built = True
    
    def forward(self, input:np.ndarray, *, training:bool=True)->np.ndarray:
        """
        input: (input_size, batch_size)
        output: (output_size, batch_size)
        """
        assert self.built
        if len(input.shape) == 1:
            # if the input is a single sample as a 1d row, reshape it as a column
            input = input.reshape(input.shape[0], 1)
        elif input.shape[1] == self.input_size:
            # transforming the features as columns
            input = input.T

        assert input.shape[0] == self.input_size

        next = input
        for layer in self.children:
            if isinstance(layer, Dropout) and not training:
                continue
            next = layer(next)
        self.output = next
        return self.output
    
    def backward(self, target:np.ndarray)->None:
        """
        output: (classes, batch_size)
        target: (batch_size, 1) or (batch_size, classes)[one hot encoded] 
        """
        # print("\nIN BACKWARD")
        if len(target.shape) == 1:
            target = one_hot(target, classes = self.output_size)
        elif target.shape == (self.batch_size, self.output_size):
            # if the target is a one hot encoded matrix, transform it to a column
            target = target.T
        assert target.shape == self.output.shape

        # the final layer has cross entropy with one hot and softmax activation
        # we can do a shortcut and find the derivative of loss w.r.t z instead of a when this is the case
        delta = self.output - target   # target has to be one hot encoded
        for i, layer in enumerate(reversed(self.children)):
            if i == 0: continue
            delta = layer.backward(delta, self.learning_rate)

    def train(self, X:np.ndarray, y:np.ndarray)->None:
        # train by mini batch
        for _ in tqdm(range(self.epochs)):
            for i in range(0, X.shape[0], self.batch_size):
                self.forward(X[i:i+self.batch_size])
                self.backward(y[i:i+self.batch_size])



    def predict(self, X:np.ndarray)->np.ndarray:
        return np.argmax(self.forward(X, training=False), axis=0)
            

    def accuracy(self, predictions:np.ndarray, target:np.ndarray)->np.float64:
        """
        output: (classes, batch_size)
        target: (batch_size, )
        """
        return np.mean(predictions == target)

    def score(self, X:np.ndarray, y:np.ndarray)->tuple:
        output = self.predict(X)
        accuracy = self.accuracy(output, y)
        loss = self.loss(self.output, y)
        # confusion matrix
        confusion = np.zeros((self.output_size, self.output_size))
        for i in range(len(y)):
            confusion[output[i], y[i]] += 1
        confusion = confusion.astype(int)
        return accuracy, loss, confusion
    
    def macro_f1(self, X:np.ndarray, y:np.ndarray)->np.float64:
        output = self.predict(X)
        
        confusion = np.zeros((self.output_size, self.output_size))
        for i in range(len(y)):
            confusion[output[i], y[i]] += 1
        confusion = confusion.astype(int)
        precision = np.zeros(self.output_size)
        recall = np.zeros(self.output_size)
        for i in range(self.output_size):
            precision[i] = confusion[i, i] / np.sum(confusion[i, :])
            recall[i] = confusion[i, i] / np.sum(confusion[:, i])
        f1 = 2 * precision * recall / (precision + recall)
        return np.mean(f1)

    def export(self, filepath:str)->None:
        assert self.built
        layers = []
        for layer in self.children:
            layer_info = {}
            layer_info['name'] = layer.name
            layer_info['input_size'] = layer.input_size
            layer_info['output_size'] = layer.output_size
            if isinstance(layer, DenseLayer):
                # create a dictionary
                layer_info['weights'] = layer.weights
                layer_info['bias'] = layer.bias
            if isinstance(layer, Dropout):
                layer_info['keep_prob'] = layer.keep_prob
            layers.append(layer_info)
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pickle.dump(layers, f)


    def describe(self)->None:
        assert self.built
        print("learning rate:", self.learning_rate)
        print("batch size:", self.batch_size)
        print("epochs:", self.epochs)
        print()
        for layer in self.children:
            print(layer)
        print(self.loss)
        print()
    
    # def graphs(self, model_number):
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(self.training_losses, label="Training Loss")
    #     plt.plot(self.validation_losses, label="Validation Loss")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.legend()
    #     plt.savefig('offline-3-fnn/report/images/'+model_number+'/loss.png')

    #     plt.figure(figsize=(8, 6))
    #     plt.plot(self.training_accuracies, label="Training Accuracy")
    #     plt.plot(self.validation_accuracies, label="Validation Accuracy")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Accuracy")
    #     plt.legend()
    #     plt.savefig('offline-3-fnn/report/images/'+model_number+'/accuracy.png')


def load_model(filepath:str)->FNN:
    layer_infos = pickle.load(filepath)
    layers = []
    model = FNN(layer_infos[0]['input_size'], layer_infos[-1]['output_size'])
    for layer_info in layer_infos:
        if layer_info['name'] == 'Dense':
            layers.append(DenseLayer(layer_info['input_size'], layer_info['output_size']))
            layers[-1].weights = layer_info['weights']
            layers[-1].bias = layer_info['bias']
        if layer_info['name'] == 'ReLU':
            layers.append(ReLU())
        if layer_info['name'] == 'Sigmoid':
            layers.append(Sigmoid())
        if layer_info['name'] == 'Tanh':
            layers.append(Tanh())
        if layer_info['name'] == 'Dropout':
            layers.append(Dropout(layer_info['keep_prob']))
        if layer_info['name'] == 'Softmax':
            layers.append(Softmax())
    model.sequential(*layers)
    model.built = True
    return model

def export_model(model:FNN, filepath:str)->None:
    # dump only the weights and biases
    assert model.built
    layers = []
    for layer in model.children:
        layer_info = {}
        layer_info['name'] = layer.name
        layer_info['input_size'] = layer.input_size
        layer_info['output_size'] = layer.output_size
        if isinstance(layer, DenseLayer):
            # create a dictionary
            layer_info['weights'] = layer.weights
            layer_info['bias'] = layer.bias
        if isinstance(layer, Dropout):
            layer_info['keep_prob'] = layer.keep_prob
        layers.append(layer_info)

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, 'wb') as f:
        pickle.dump(layers, f)