from layers import LinearLayer
from activation_functions import SigmoidActivation
from losses import MeanSquareLoss


class NeuralNetwork:

    """
    A simple neural network.
    """

    def __init__(self):

        # Computation graph is a doubly-linked list
        self.head = None
        self.tail = None

        # Record of which layers are output and loss
        self.output_layer = None
        self.loss_layer = None

    def _add_layer(self, new_layer):

        """ Helper function for adding a new layer to the computation graph"""

        if self.tail:
            self.tail.next = new_layer
            new_layer.prev = self.tail
        else:
            self.head = self.tail = new_layer

        self.tail = new_layer

    def add_input_layer(self, n_in, n_out):

        new_layer = LinearLayer(n_in, n_out)
        self._add_layer(new_layer)

    def add_output_layer(self, n_out):

        n_in = self.tail.output_size
        new_layer = LinearLayer(n_in, n_out)
        self._add_layer(new_layer)
        self.output_layer = new_layer

    def add_loss_layer(self):

        new_layer = MeanSquareLoss()
        self._add_layer(new_layer)
        self.loss_layer = new_layer

    def add_sigmoid_activation(self):

        n = self.tail.output_size
        new_layer = SigmoidActivation(n)
        self._add_layer(new_layer)

    def forward_pass(self, x):

        """
        Performs forward pass through neural network

        x: 2d numpy array.  Input to first layer.
        """

        self.head.input = x

        layer = self.head
        while layer.next:
            layer.forward_pass()
            layer.next.input = layer.output  # propagate forward pass
            layer = layer.next

        # forward pass on final layer
        layer.forward_pass()

    def backwards_pass(self):

        """
        Performs the backwards pass of the neural network.
        """

        layer = self.loss_layer
        while layer.prev:
            layer.backward_pass()
            layer.prev.dL_dout = layer.dL_din
            layer = layer.prev

        # backwards pass on first layer
        layer.backward_pass()

    def calc_gradients(self):

        """
        Calculates the parameter gradients of the neural network

        """

        layer = self.head

        while True:
            layer.calc_param_grads()

            if not layer.next:
                break

            layer = layer.next

    def update_params(self, learn_rate):

        """
        Updates the parameters of the neural network

        learn_rate: Float. Learning rate for neural network parameter updates
        """

        layer = self.head

        while True:
            layer.update_params(learn_rate)

            if not layer.next:
                break

            layer = layer.next

    def training_step(self, x, y, learn_rate):

        self.loss_layer.y = y

        self.forward_pass(x)
        self.backwards_pass()
        self.calc_gradients()
        self.update_params(learn_rate)