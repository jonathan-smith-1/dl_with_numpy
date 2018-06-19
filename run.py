"""Demonstration of building and training a neural network."""

import numpy as np
from dl_with_numpy.network import NeuralNetwork


def main():
    """
    Create a neural network and fit it to a dummy dataset.

    Returns:
        Nothing

    """
    x_train = np.array([[1., 2., 3.],
                        [8., 6., 3]])

    w_generate = np.array([[2.],
                           [3.],
                           [1.]])

    b_generate = 4.

    y_train = np.matmul(x_train, w_generate) + b_generate

    # Build neural network
    network = NeuralNetwork()

    network.add_input_layer(x_train.shape[1], n_out=6)
    network.add_sigmoid_activation()
    network.add_linear_layer(n_out=4)
    network.add_sigmoid_activation()
    network.add_output_layer(n_out=1)
    network.add_mse_loss_layer()

    # Train the neural network
    steps = 500

    for step in range(steps):

        network.training_step(x_train, y_train, learn_rate=0.01)
        print('Step: {}   Loss: {:.2f}'
              .format(step, network.loss_layer.mean_sq_loss))


main()
