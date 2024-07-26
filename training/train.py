import torchvision.datasets as ds
from torchvision import transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import datetime


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tworch

if __name__ == "__main__":
    train_validation_dataset = load_digits()
    print("dataset loaded")
    input_size = 64        # 8 * 8
    output_size = 10        # 10 digits
    X_train, X_validation, y_train, y_validation = train_test_split(train_validation_dataset.data, train_validation_dataset.target, test_size=0.2, random_state=42)

    learning_rate = 5e-4
    batch_size = 1024
    epochs = 100
    
    # the filepath should have the current datetime in the end
    model_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filepath = "models/model-"+model_time+".pkl"
    modelpath = "models/model-"+model_time+".pkl"
    
    model = tworch.network.FNN(input_size, output_size, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
    model.sequential(tworch.layer.DenseLayer(input_size, 128, initializer=tworch.initializer.He()),
                    tworch.layer.ReLU(),
                    tworch.layer.Dropout(0.8),
                    tworch.layer.DenseLayer(128, output_size, initializer=tworch.initializer.Xavier()),
                    tworch.layer.Softmax())
    
    
    model.describe()
    model.train(X_train, y_train)
    model.export(filepath)
    with open(modelpath, 'rb') as f:
        model = tworch.network.load_model(f)
    model.describe()
    
    training_accuracy, training_loss, training_confusion = model.score(X_train, y_train)
    validation_accuracy, validation_loss, validation_confusion = model.score(X_validation, y_validation)
    validation_macro_f1 = model.macro_f1(X_validation, y_validation)

    print(f'training accuracy:\t {training_accuracy*100:.2f}%')
    print(f'validation accuracy:\t {validation_accuracy*100:.2f}%')
    print(f'training loss:\t\t {training_loss:.2f}')
    print(f'validation loss:\t {validation_loss:.2f}')
    print(f'validation macro f1:\t {validation_macro_f1:.2f}')
