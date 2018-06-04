import numpy as np
from layers import Layer


class SigmoidActivation(Layer):
    """
    Sigmoid activation layer of neural network

    n: Integer. Size of layer
    """

    def __init__(self, n):
        super(SigmoidActivation, self).__init__(n_in=n, n_out=n)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def forward_pass(self):
        self.output = self.sigmoid(self.input)

    def backward_pass(self):
        self.dL_din = self.dL_dout * self.sigmoid_deriv(self.input)
