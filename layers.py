import numpy as np


class Layer:
    """
    Base class for a single layer of neural network.
    """

    def __init__(self, n_in, n_out):

        self.input = None
        self.output = None

        self.input_size = n_in
        self.output_size = n_out

        self.dL_din = None  # derivative of loss w.r.t. input
        self.dL_dout = None  # derivative of loss w.r.t. output

        self.next = None  # next node in the computation graph
        self.prev = None  # previous node in the computation graph

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

    def calc_param_grads(self):
        pass

    def update_params(self, lr):
        pass


class LinearLayer(Layer):
    """
    Linear layer of neural network.

    n_in: Integer. Size of input to layer
    n_out: Integer. Size of output of layer
    seed: Integer. Random seed for initialisation of layer's parameters
    """

    def __init__(self, n_in, n_out, seed=0):
        super(LinearLayer, self).__init__(n_in=n_in, n_out=n_out)

        np.random.seed(seed)
        self.W = np.random.normal(0.0, n_in ** -0.5, (n_in, n_out))  # parameters of the linear layer
        self.dL_dW = None  # derivative of loss w.r.t. parameters W

    def forward_pass(self):
        self.output = np.dot(self.input, self.W)

    def backward_pass(self):
        self.dL_din = np.dot(self.dL_dout, self.W.T)

    def calc_param_grads(self):
        self.dL_dW = np.dot(self.input.T, self.dL_dout)

    def update_params(self, lr):
        self.W -= self.dL_dW * lr
