import numpy as np
import matplotlib.pyplot as plt
import pickle


class Model:
   def __init__(self, network):
      self.network = network # a list of layer Objects like Dense
      self.costs = []

   def mse(self, y_true, y_pred):
      return np.mean(np.power(y_true - y_pred, 2))

   def mse_dash(self, y_true, y_pred):
      return 2 * (y_pred - y_true) / np.size(y_true)

   def train(self, X, Y, epochs=500):
      for epoch in range(epochs):
         cost = 0
         for x, y in zip(X, Y):

            output = x
            for layer in self.network:
               output = layer.forward_prop(output)

            cost += self.mse(y, output)
            
            dcy = self.mse_dash(y, output)
            for layer in self.network[::-1]:
               dcy = layer.backward_prop(dcy)

         cost /= len(X)
         self.costs.append(cost)

         print(f'Epoch : {epoch} || Cost : {cost}')

   def predict(self, x):
      output = x
      for layer in self.network:
         output = layer.forward_prop(output)

      return output

   def plot_costs(self):
      plt.plot(self.costs)
      plt.xlabel('Epochs')
      plt.ylabel('Cost')
      plt.show()


   def save(self, file_name):
      with open(file_name, 'wb') as file:
         pickle.dump(self, file)

def load_model(file_name):
   with open(file_name, 'rb') as file:
      model_file = pickle.load(file)
   return model_file
