#!/bin/bash

# python vis.py --dataset_name 'down-up-down' --layers 2 --neurons 10 --activation 'Tanh' 
# python vis.py --dataset_name 'down-up-down' --layers 2 --neurons 10 --activation 'ReLU'
# python vis.py --dataset_name 'down-up-down' --layers 2 --neurons 10 --activation 'Sigmoid' 
python vis.py --dataset_name 'gaussian' --layers 5 --neurons 10 --activation 'ReLU' --epochs 150
python vis.py --dataset_name 'gaussian' --layers 5 --neurons 10 --activation 'Tanh' --epochs 150
python vis.py --dataset_name 'gaussian' --layers 5 --neurons 10 --activation 'Sigmoid' --epochs 150
python vis.py --dataset_name 'gaussian' --layers 5 --neurons 10 --activation 'Softplus' --epochs 150

