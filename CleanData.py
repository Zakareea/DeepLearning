import numpy as np

#Flatten an image
def flatten(array):
	shape = array.shape[2] * array.shape[1]
	arr_flattened = array.reshape(array.shape[0], shape, 1)
	return arr_flattened

#encode labels
def one_hot_encoding(labels):
   encoded_labels = []
   for y in labels:
      zeros = np.zeros((10))
      zeros[y] = 1
      encoded_labels.append(zeros)

   return np.array(encoded_labels)
