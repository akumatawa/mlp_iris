import numpy as np

import functions as f

# Setup the parameters
input_layer_size = 4
hidden_layer_size = 10
num_labels = 3

lam = 0
learning_rate = 0.2
epochs = 500
test_size = 0.3

data = np.load("iris_processed.npy", allow_pickle=True)
np.random.shuffle(data)
train_size = int(data.shape[0] * (1 - test_size))
training, test = data[:train_size, :], data[train_size:, :]

x_train = training[:, 0:4]
x_test = test[:, 0:4]
y_train = training[:, 4]
y_test = test[:, 4]

initial_theta = f.initialize_model(input_layer_size, hidden_layer_size, num_labels)

theta, error = f.train(x_train, x_test, y_train, y_test, initial_theta, lam, learning_rate, epochs)

f.learning_curve(epochs, error)
