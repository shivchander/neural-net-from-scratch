#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Utils: Single Hidden layer DenseNN
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold


def load_data():
    data = pd.read_csv('dataset/data_banknote_authentication.txt', header=None)
    y = data[4]
    X = data.drop(4, 1)

    return X, y.to_numpy().reshape(y.shape[0], 1)

