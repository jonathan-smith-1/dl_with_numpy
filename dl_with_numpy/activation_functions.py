import numpy as np
from dl_with_numpy.layer import Layer


class SigmoidActivation(Layer):

    """
    Sigmoid activation layer of a neural network.
    """

    def __init__(self, n):

        """
        Create activation layer.

        Args:
            n (integer): Size of input and output data.  This layer accepts inputs with dimension [batch_size, n]
                         and produces an output of the same dimensions.
        """
        super(SigmoidActivation, self).__init__(n_in=n, n_out=n)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def forward_pass(self):
        self.output = self.sigmoid(self.input)

    def backward_pass(self):
        self.dL_din = self.dL_dout * self.sigmoid_deriv(self.input)
