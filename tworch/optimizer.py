import numpy as np

class Optimizer():
    def __init__(self):
        pass
    def __str__(self) -> str:
        return "Optimizer"
    def update(self, weights:np.ndarray, delta_weights:np.ndarray, learning_rate:float)->np.ndarray:
        pass

class SGD(Optimizer):
    def __init__(self):
        pass
    def __str__(self) -> str:
        return "SGD"
    def update(self, weights:np.ndarray, delta_weights:np.ndarray, learning_rate:float)->np.ndarray:
        return weights - learning_rate * delta_weights
    
class Momentum(Optimizer):
    def __init__(self, input_size:int, output_size:int, beta:float=0.9):
        self.beta = beta
        self.v = np.zeros((output_size, input_size))
    def __str__(self) -> str:
        return "Momentum: " + str(self.beta)
    def update(self, weights:np.ndarray, delta_weights:np.ndarray, learning_rate:float)->np.ndarray:
        self.v = self.beta * self.v + (1 - self.beta) * delta_weights
        weights -= learning_rate * self.v
        return weights
    
class RMSProp(Optimizer):
    def __init__(self, input_size:int, output_size:int, beta:float=0.9, epsilon:float=1e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.sum_of_squares = np.zeros((output_size, input_size))
    def __str__(self) -> str:
        return "RMSProp: " + str(self.beta)
    def update(self, weights:np.ndarray, delta_weights:np.ndarray, learning_rate:float)->np.ndarray:
        self.sum_of_squares = self.beta * self.sum_of_squares + (1 - self.beta) * delta_weights**2
        weights -= learning_rate * delta_weights / (np.sqrt(self.sum_of_squares) + self.epsilon)
        return weights    


class Adam(Optimizer):
    def __init__(self, input_size:int, output_size:int, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros((output_size, input_size))
        self.v = np.zeros((output_size, input_size))
        self.t = 1
    def __str__(self) -> str:
        return "Adam: " + str(self.beta1) + " " + str(self.beta2) + " " + str(self.epsilon)
    def update(self, weights:np.ndarray, delta_weights:np.ndarray, learning_rate:float)->np.ndarray:
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta_weights
        self.v = self.beta2 * self.v + (1 - self.beta2) * delta_weights**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.t += 1
        return weights