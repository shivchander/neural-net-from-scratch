#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Main: Single Hidden layer DenseNN
'''

from utils import *
from model import DenseNN

# load data
data_X, data_y = load_data()

# split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y , test_size=0.2)

model = DenseNN(n_x=4, n_h=4, n_y=1, hidden_activation='tanh')
model.fit(X_train, y_train, alpha=0.001, epochs=1000)
y_preds = model.predict(X_test)
print(balanced_accuracy_score(y_test, y_preds))
