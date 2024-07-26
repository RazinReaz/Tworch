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
- **[Feed Forward Neural Network](https://github.com/RazinReaz/Tworch/blob/main/docs/fnn.md)**



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
