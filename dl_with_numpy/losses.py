import numpy as np
from dl_with_numpy.layers import Layer


class MeanSquareLoss(Layer):
    """
    Mean square loss layer for neural network.
    """

    def __init__(self):

        """
        Create mean square loss layer
        """

        super(MeanSquareLoss, self).__init__(n_in=0, n_out=0)  # Dimension of loss layer is not useful, set to 0

        self.dL_dout = 1  # output of this layer is the loss, so this derivative w.r.t. loss is 1
        self.mean_sq_loss = None
        self.y = None  # true output

    def forward_pass(self):
        self.mean_sq_loss = 0.5 * np.mean(np.square(self.input - self.y))

    def backward_pass(self):
        self.dL_din = (self.input - self.y) / self.y.size
