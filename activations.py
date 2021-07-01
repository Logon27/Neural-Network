from activation import Activation
import numpy as np

def softmax_grad(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

#Activation Function I implemented
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        super().__init__(sigmoid, sigmoid_prime)

class Relu(Activation):
    def __init__(self):
        #relu = lambda x: np.maximum(x, 0)
        relu = lambda x: np.where(x > 0, dout, 0)
        #relu_prime = lambda x: np.heaviside(x, 1)
        relu_prime = lambda x: 1. * (x > 0)
        super().__init__(relu, relu_prime)

class Leaky_Relu(Activation):
    def __init__(self):
        #In progress may not work
        alpha = 0.1
        leaky_relu = lambda x: np.maximum(alpha*x, x)
        #might need to change astype
        leaky_relu_prime = lambda x: 0
        super().__init__(leaky_relu, leaky_relu_prime)

class Sin(Activation):
    def __init__(self):
        sin = lambda x: np.sin(x)
        #relu_prime = lambda x: np.heaviside(x, 1)
        sin_prime = lambda x: np.cos(x)
        super().__init__(sin, sin_prime)
