import numpy as np
import cv2
from scipy import signal

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

class ZConv:
	def __init__(self, kernel_num, kernel_size, input_shape, activation):
		self.kernel_num = kernel_num
		self.kernel_size = kernel_size
		self.input_shape = input_shape
		self.activation = activation
		self.depth = self.input_shape[2]
		self.output_shape = (self.kernel_num, self.input_shape[0] - self.kernel_size[0] + 1, self.input_shape[1] - self.kernel_size[1] + 1)
		self.kernel_shape = (self.kernel_num, self.kernel_size[0], self.kernel_size[1], self.depth)
		self.kernels = np.random.randn(*self.kernel_shape)
		self.biases = np.random.randn(*self.output_shape)

	def ReLU(self, x):
		return np.maximum(0, x)
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def forward_prop(self, x):
		self.input = x
		self.output = np.copy(self.biases)
		for d in range(self.kernel_num):
			for i, k in zip(cv2.split(self.input), cv2.split(self.kernels[d])):
				self.output[d] += signal.correlate2d(i, k, "valid")

		if self.activation == "relu":
			return self.ReLU(self.output)
		if self.activation == "sigmoid":
			return self.sigmoid(self.output)
		if self.activation == "softmax":
			return self.softmax(self.output)

