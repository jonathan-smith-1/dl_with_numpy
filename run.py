import numpy as np
#from my_answers import LinearLayer, SigmoidActivation, MeanSquareLoss, NeuralNetwork
from network import NeuralNetwork


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

iterations = 2000
learning_rate = 1.5
hidden_nodes = 6
output_nodes = 1

N_i = x_train.shape[1]
network = NeuralNetwork([N_i, hidden_nodes, output_nodes])


for step in range(250):
    # Printing out the training progress
    network.training_step(x_train, y_train, learn_rate=0.01)

    network.forward_pass(x_train, y_train)  # TODO - Remove the need for y_train here
    train_loss = 2 * network.loss_layer.mean_sq_loss  # mean_sq_loss attribute is 0.5*MSE
    print('Step: {}   Loss: {:.2f}'.format(step, train_loss))
