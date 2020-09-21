import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def randomization(n):
    return np.random.random([n, 1])


def operations(h, w):
    mat1 = np.random.random([h, w])
    mat2 = np.random.random([h, w])

    s = mat1 + mat2

    return [mat1, mat2, s]


def norm(A, B):
    s = A + B
    return np.linalg.norm(s)


def neural_network(inputs, weights):
    return np.array([[np.tanh(np.sum(inputs * weights))]])

weights = np.array([[0.24755791], [0.24488441]])
inputs = np.array([[0.40299008], [0.74212644]])

print(neural_network(inputs, weights))

print(neural_network(inputs, weights).size)


def scalar_function(x, y):
    if np.greater_equal(x, y):
        return np.dot(x, y)
    else:
        return np.divide(x, y)
