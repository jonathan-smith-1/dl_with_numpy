import pytest
import numpy as np
from dl_with_numpy.network import NeuralNetwork


def test_train_empty_network():

    """
    WHEN:
        The neural network has no layers

    AND:
        The training_step method is called

    WHAT SHOULD HAPPEN:
        A ValueError Exception should be raised.

    Returns:
        Nothing

    """

    with pytest.raises(ValueError):

        network = NeuralNetwork()

        x = np.array([[0.]])
        y = np.array([[1.]])

        network.training_step(x, y, learn_rate=0.01)



def test_multiple_input_layers():

    assert False


def test_no_input_layer():

    assert False


def test_multiple_loss_layers():

    assert False


def test_no_loss_layer():

    assert False


def test_contiguous_activation_layers():

    assert False
