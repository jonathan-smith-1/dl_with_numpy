import pytest
import numpy as np
from dl_with_numpy.network import NeuralNetwork


def test_basic_operation():

    """
    WHEN:
        A basic neural network is constructed

    AND:
        The training_step method is called

    REQUIREMENT:
        No exceptions shall be raised.


    Returns:
        Nothing

    """

    x_train = np.array([[1., 2., 3.],
                        [8., 6., 3]])

    y_train = np.array([[1.],
                        [2.]])

    # Build neural network
    network = NeuralNetwork()

    network.add_input_layer(x_train.shape[1], n_out=6)
    network.add_sigmoid_activation()
    network.add_linear_layer(n_out=4)
    network.add_sigmoid_activation()
    network.add_output_layer(n_out=1)
    network.add_mse_loss_layer()

    # Train neural network
    iterations = 1

    for step in range(iterations):
        network.training_step(x_train, y_train, learn_rate=0.01)


def test_train_empty_network():

    """
    WHEN:
        The neural network has no layers

    AND:
        The training_step method is called

    REQUIREMENT:
        A ValueError Exception shall be raised


    Returns:
        Nothing

    """

    # TODO - Remove this method if it is dominated by test_no_input_layer or test_no_loss_layer

    with pytest.raises(ValueError):

        network = NeuralNetwork()

        x = np.array([[0.]])
        y = np.array([[1.]])

        network.training_step(x, y, learn_rate=0.01)


def test_multiple_input_layers():

    """
    WHEN:
        The neural network already has an input layer

    AND:
        The add_input_layer method is called

    REQUIREMENT:
        A ValueError Exception shall be raised


    Returns:
        Nothing

    """

    with pytest.raises(ValueError):

        network = NeuralNetwork()

        network.add_input_layer(n_in=2, n_out=2)
        network.add_input_layer(n_in=2, n_out=2)


def test_add_linear_layer_without_input_layer():

    """
    WHEN:
        The neural network has no input layer

    AND:
        The add_linear_layer method is called

    REQUIREMENT:
        A ValueError Exception shall be raised


    Returns:
        Nothing

    """

    with pytest.raises(ValueError):

        network = NeuralNetwork()

        network.add_linear_layer(n_out=2)


def test_add_output_layer_without_input_layer():
    """
    WHEN:
        The neural network has no input layer

    AND:
        The add_output_layer method is called

    REQUIREMENT:
        A ValueError Exception shall be raised


    Returns:
        Nothing

    """

    with pytest.raises(ValueError):

        network = NeuralNetwork()

        network.add_output_layer(n_out=2)


def test_add_mse_loss_layer_without_input_layer():
    """
    WHEN:
        The neural network has no input layer

    AND:
        The add_mse_loss_layer method is called

    REQUIREMENT:
        A ValueError Exception shall be raised


    Returns:
        Nothing

    """

    with pytest.raises(ValueError):

        network = NeuralNetwork()

        network.add_mse_loss_layer()


def test_add_sigmoid_activation_layer_without_input_layer():
    """
    WHEN:
        The neural network has no input layer

    AND:
        The add_activation_layer method is called

    REQUIREMENT:
        A ValueError Exception shall be raised


    Returns:
        Nothing

    """

    with pytest.raises(ValueError):

        network = NeuralNetwork()

        network.add_sigmoid_activation()


def test_two_mse_loss_layers():

    """
    WHEN:
        The neural network already has a loss layer

    AND:
        The add_mse_loss_layer method is called

    REQUIREMENT:
        A ValueError Exception shall be raised


    Returns:
        Nothing

    """

    with pytest.raises(ValueError):

        network = NeuralNetwork()

        network.add_input_layer(n_in=2, n_out=2)
        network.add_mse_loss_layer()
        network.add_mse_loss_layer()


def test_no_loss_layer():

    """
    WHEN:
        The neural network has no loss layer

    AND:
        The training_step method is called

    REQUIREMENT:
        A ValueError Exception shall be raised


    Returns:
        Nothing

    """

    with pytest.raises(ValueError):

        # Mini dataset
        x_train = np.array([[1., 2., 3.],
                            [8., 6., 3]])

        y_train = np.array([[2.],
                            [3.]])

        # Build neural network
        network = NeuralNetwork()

        network.add_input_layer(x_train.shape[1], n_out=3)
        network.add_output_layer(n_out=1)

        network.training_step(x_train, y_train, learn_rate=0.01)


def test_contiguous_activation_layers():

    """
    WHEN:
        The neural network has two contiguous sigmoid activation layers

    AND:
        The training_step method is called

    REQUIREMENT:
        No exception shall be raised


    Returns:
        Nothing

    """

    network = NeuralNetwork()

    network.add_input_layer(n_in=2, n_out=3)
    network.add_sigmoid_activation()
    network.add_sigmoid_activation()
