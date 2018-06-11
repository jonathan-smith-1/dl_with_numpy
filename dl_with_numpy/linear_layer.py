from dl_with_numpy.layer import Layer
import numpy as np


class LinearLayer(Layer):

    """
    A linear layer for a neural network.
    """

    def __init__(self, n_in, n_out, seed=0):

        """

        Args:
            n_in (integer): Size of input to this layer.  This layer accepts inputs with dimension [batch_size, n_in].

            n_out (integer): Size of output of this layer.  This layer creates outputs with dimension
                             [batch_size, n_out]

            seed (integer): Random seed for initialising the linear layer's parameters.
        """

        super(LinearLayer, self).__init__(n_in=n_in, n_out=n_out)

        np.random.seed(seed)

        self.W = np.random.normal(0.0, n_in ** -0.5, (n_in, n_out))  # Initialise parameters of the linear layer
                                                                     # with fan-in approach.
        self.dL_dW = None  # derivative of loss w.r.t. parameters W

    def forward_pass(self):
        self.output = np.dot(self.input, self.W)

    def backward_pass(self):
        self.dL_din = np.dot(self.dL_dout, self.W.T)

    def calc_param_grads(self):
        self.dL_dW = np.dot(self.input.T, self.dL_dout)

    def update_params(self, lr):
        self.W -= self.dL_dW * lr
