from Layers import Dense
from CleanData import flatten, one_hot_encoding
from Model import Model, load_model
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd

#import your mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#prepare you dataset
def data_preparation(X, Y):
	X = X[:2000]
	X = flatten(X) / 255

	Y = Y[:2000]
	Y = one_hot_encoding(Y)
	Y = np.reshape(Y, (2000, 10, 1))

	return X, Y

X, Y = data_preparation(x_train, y_train)

#initialize your neural network
network = [Dense(784, 8, activation='relu'), Dense(8, 10, activation='softmax')]
#model = Model(network)

#train the model
#model.train(X, Y)

#save the model
#mode.save('mnist_model')

#load the model
model = load_model('mnist_model')
model.plot_costs()