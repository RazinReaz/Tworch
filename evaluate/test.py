import torchvision.datasets as ds
from torchvision import transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import classes.utils as utils
import classes.network as network
import classes.Layer as Layer
import classes.Initializer as Initializer
import classes.Loss as Loss
    

if __name__ == "__main__":
    
    independent_test_set = ds.EMNIST(root='./offline-3-fnn/data', split='letters',
                              train=False,
                              transform=transforms.ToTensor(),
                              download = False)
    print("test dataset loaded")
   
    X_test = np.array([sample[0].numpy().flatten() for sample in independent_test_set])
    y_test = np.array([sample[1] for sample in independent_test_set]) - 1
    
    print("test dataset converted to numpy arrays")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=int, help="model number to evaluate")
    args = parser.parse_args()

    modelpath = args.modelpath
    
    with open(modelpath, 'rb') as f:
        model = network.create_model(f)
    print("model loaded from", modelpath)
    model.describe()

    test_accuracy, test_loss, test_confusion = model.score(X_test, y_test)
    test_macro_f1 = model.macro_f1(X_test, y_test)
    print("test accuracy:\t\t", test_accuracy*100, "%")
    print("test loss:\t\t", test_loss)
    print("test macro f1:\t\t", test_macro_f1)

    characters = [chr(i+97) for i in range(26)]
    utils.confusion_heatmap(test_confusion, labels=characters, title="Test Confusion Matrix", save_path="offline-3-fnn/report/images/"+modelpath+"/test_confusion.png")
