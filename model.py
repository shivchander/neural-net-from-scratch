#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Model Definition: Single Hidden layer DenseNN
'''
import numpy as np


class DenseNN:
    def __init__(self, n_x, n_h, n_y, hidden_activation='tanh'):

        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros(shape=(n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros(shape=(n_y, 1))

        self.parameters = {"W1": W1,
                           "b1": b1,
                           "W2": W2,
                           "b2": b2}
        self.hidden_activation = hidden_activation

    def activation_forward(self, A_prev, W, b, activation):

        def linear_forward(A, W, b):
            Z = np.dot(W, A) + b
            cache = (A, W, b)
            return Z, cache

        def sigmoid(Z):
            A = 1 / (1 + np.exp(-Z))
            cache = Z

            return A, cache

        def tanh(Z):
            A = np.exp(Z) - np.exp(-Z) / np.exp(Z) + np.exp(-Z)
            cache = Z

            return A, cache

        if activation == "sigmoid":
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        if activation == "tanh":
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = tanh(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_propagation(self, X):
        caches = []
        A = X
        A_prev = A
        A, cache = self.activation_forward(A_prev, self.parameters['W1'],
                                           self.parameters['b1'], activation=self.hidden_activation)
        caches.append(A)

        AL, cache = self.activation_forward(A, self.parameters['W2'],
                                            self.parameters['b2'], activation='sigmoid')
        caches.append(AL)
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        return cost

    def backward_propagation(self, cache, X, Y):
        m = X.shape[1]

        W1 = self.parameters['W1']
        W2 = self.parameters['W2']

        A1 = cache[0]
        A2 = cache[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        if self.hidden_activation == 'sigmoid':
            dZ1 = np.multiply(np.dot(W2.T, dZ2), np.multiply(A1, 1 - A1))
        if self.hidden_activation == 'tanh':
            dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

    def update_parameters(self, grads, learning_rate=1e-3):

        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']

        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def fit(self, X, Y, epochs=1000, alpha=1e-3, verbose=True):

        X = X.T
        Y = Y.T
        for i in range(0, epochs):
            A2, cache = self.forward_propagation(X)
            cost = self.compute_cost(A2, Y)
            grads = self.backward_propagation(cache, X, Y)
            self.parameters = self.update_parameters(grads, alpha)

            if verbose and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return self.parameters

    def predict(self, X):
        X = X.T
        a, caches = self.forward_propagation(X)
        predictions = a.T
        predictions[predictions <= 0.5] = 0
        predictions[predictions > 0.5] = 1

        return predictions
