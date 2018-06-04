import numpy as np
from layers import Layer


class MeanSquareLoss(Layer):
    """
    Mean square loss function

    """

    def __init__(self):
        super(MeanSquareLoss, self).__init__(n_in=None, n_out=None)

        self.dL_dout = 1  # output of this layer is the loss, so this derivative is 1
        self.mean_sq_loss = None
        self.y = None  # true output

    def forward_pass(self):
        self.mean_sq_loss = 0.5 * np.mean(np.square(self.input - self.y))

    def backward_pass(self):
        self.dL_din = (self.input - self.y) / self.y.size