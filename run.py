import numpy as np
#from my_answers import LinearLayer, SigmoidActivation, MeanSquareLoss, NeuralNetwork
from network import NeuralNetwork
from layers import LinearLayer
from activation_functions import SigmoidActivation
from losses import MeanSquareLoss

# Construct the mini dataset for testing
x_train = np.array([[1., 2., 3.],
                    [8., 6., 3]])

W_true = np.array([[2.],
                   [3.],
                   [1.]])

b_true = 4.

y_train = np.matmul(x_train, W_true) + b_true

debug = 0

# Build neural network

iterations = 250
learning_rate = 1.5
hidden_nodes = 6
output_nodes = 1

N_i = x_train.shape[1]

network = NeuralNetwork()

network.add_input_layer(N_i, hidden_nodes)
network.add_sigmoid_activation()


network.add_output_layer(n_out=1)

network.add_loss_layer()


for step in range(iterations):
    # Printing out the training progress
    network.training_step(x_train, y_train, learn_rate=0.01)

    network.forward_pass(x_train)
    train_loss = 2 * network.loss_layer.mean_sq_loss  # mean_sq_loss attribute is 0.5*MSE
    print('Step: {}   Loss: {:.2f}'.format(step, train_loss))
