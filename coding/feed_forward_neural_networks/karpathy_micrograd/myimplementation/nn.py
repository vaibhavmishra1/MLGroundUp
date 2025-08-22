import numpy as np 
import matplotlib.pyplot as plt
import math
import random
from engine import Value

class Neuron:
    def __init__(self, input_dim, activation_function="tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(input_dim)]
        self.b = Value(random.uniform(-1, 1))
        self.activation_function = activation_function

    def __call__(self, x):
        if self.activation_function == "tanh":
            return (sum(w * x for w, x in zip(self.w, x)) + self.b).tanh()
    

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"Neuron(w={self.w}, b={self.b})"
    
    
    
class Layer:
    def __init__(self, input_dim, output_dim, activation_function="tanh"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.neurons = [Neuron(input_dim, activation_function) for _ in range(output_dim)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
    
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
    

class NeuralNetwork:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = [Layer(input_dim, 8, activation_function = "tanh") ,
                       Layer(8, 4, activation_function = "tanh") ,
                       Layer(4, 2, activation_function = "tanh") ,
                       Layer(2, 1, activation_function = "tanh")]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]