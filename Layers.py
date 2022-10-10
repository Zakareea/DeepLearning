import numpy as np

class Dense:
	def __init__(self, input_size, output_size, activation):
		self.weights = np.random.randn(output_size, input_size)
		self.biases = np.random.randn(output_size, 1)
		self.activation = activation

	def ReLU(self, x):
		return np.maximum(0, x)
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def forward_prop(self, x):
		self.input = x
		weighted_sum = np.dot(self.weights, self.input) + self.biases

		if self.activation == "relu":
			return self.ReLU(weighted_sum) / 500
		if self.activation == "sigmoid":
			return self.sigmoid(weighted_sum)
		if self.activation == "softmax":
			return self.softmax(weighted_sum)
		if self.activation == None:
			return weighted_sum

	def backward_prop(self, dcy, lr=0.01):
		dcw = np.dot(dcy, self.input.T)
		dcb = dcy
		self.weights -= dcw * lr
		self.biases -= dcb * lr
		return np.dot(self.weights.T, dcy)
