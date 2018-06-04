import numpy as np
from network import NeuralNetwork


# Generate a mini dataset with two datapoints for testing.
x_train = np.array([[1., 2., 3.],
                    [8., 6., 3]])

W_gen = np.array([[2.],
                  [3.],
                  [1.]])

b_gen = 4.

y_train = np.matmul(x_train, W_gen) + b_gen


# Build neural network
network = NeuralNetwork()

network.add_input_layer(x_train.shape[1], n_out=6)
network.add_sigmoid_activation()
network.add_hidden_layer(n_out=4)
network.add_sigmoid_activation()
network.add_output_layer(n_out=1)
network.add_loss_layer()


# Train neural network
iterations = 500

for step in range(iterations):

    network.training_step(x_train, y_train, learn_rate=0.01)
    print('Step: {}   Loss: {:.2f}'.format(step, network.loss_layer.mean_sq_loss))
