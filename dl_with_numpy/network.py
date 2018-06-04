from layers import LinearLayer
from activation_functions import SigmoidActivation
from losses import MeanSquareLoss


class NeuralNetwork:

    """
    A simple neural network.

    The computation graph is recorded as a doubly-linked list, meaning that only simple network structures
    are permitted (i.e. no branching in computation graph).
    """

    def __init__(self):

        """
        Initialise the neural network.
        """

        # Computation graph is a doubly-linked list
        self.head = None   # Input end of computation graph
        self.tail = None

        # Record of which layers are output and loss
        self.output_layer = None
        self.loss_layer = None

    def _add_layer(self, new_layer):

        """
        Helper function for adding a new layer to the tail of the computation graph

        Args:
            new_layer (Layer object): Layer to add to the neural network.

        Returns:
            Nothing

        """

        if self.tail:
            self.tail.next = new_layer
            new_layer.prev = self.tail
        else:
            self.head = self.tail = new_layer

        self.tail = new_layer

    def add_input_layer(self, n_in, n_out):

        """
        Add a linear input layer to the neural network.  This must be the first layer added to the neural network.

        Args:
            n_in (integer): Size of inputs to this layer.  Inputs expected to have dimensions [batch_size, n_in]
            n_out (integer): Size of outputs of this layer.  Outputs will have dimensions [batch_size, n_out]

        Returns:
            Nothing

        """

        new_layer = LinearLayer(n_in, n_out)
        self._add_layer(new_layer)

    def add_hidden_layer(self, n_out):

        """
        Adds a linear hidden layer to the end of the neural network.  The input dimension to this layer is
        automatically taken from the current last layer in the network.

        Args:
            n_out (integer): Size of outputs of this layer.  Outputs will have dimensions [batch_size, n_out]

        Returns:
            Nothing

        """

        n_in = self.tail.output_size
        new_layer = LinearLayer(n_in, n_out)
        self._add_layer(new_layer)

    def add_output_layer(self, n_out):

        """
        Adds a linear output layer to the end of the neural network.  The input dimension to this layer is
        automatically taken from the current last layer in the network.

        The only difference between an output layer and a hidden layer is that an output layer's output can be
        accessed through NeuralNetwork's output_layer attribute.  Only one output layer is permitted.

        Args:
            n_out (integer): Size of outputs of this layer.  Outputs will have dimensions [batch_size, n_out]

        Returns:
            Nothing

        """

        n_in = self.tail.output_size
        new_layer = LinearLayer(n_in, n_out)
        self._add_layer(new_layer)
        self.output_layer = new_layer

    def add_loss_layer(self):

        """
        Adds a mean-square error loss layer to the end of the neural network.

        Returns:
            Nothing

        """

        new_layer = MeanSquareLoss()
        self._add_layer(new_layer)
        self.loss_layer = new_layer

    def add_sigmoid_activation(self):

        """
        Adds a sigmoid activation layer to the end of the neural network.

        Returns:
            Nothing

        """

        n = self.tail.output_size
        new_layer = SigmoidActivation(n)
        self._add_layer(new_layer)

    def forward_pass(self, x):

        """
        Performs a forward pass through the neural network.  Each layer's input and output values are stored in
        preparation for the backwards pass to calculate gradients.

        Args:
            x (2d Numpy array): Data input

        Returns:
            Nothing

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
        Performs the backwards pass of the neural network.  The gradients of each layer's input and output with
        respect to the loss are stored in preparation for the calculation of the parameter gradients.

        Returns:
            Nothing

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
        Calculates the gradients of all the parameters in each of the layers in the network.  The gradients are stored
        in preparation for updating the parameters.

        Returns:
            Nothing

        """

        layer = self.head

        while True:
            layer.calc_param_grads()

            if not layer.next:
                break

            layer = layer.next

    def update_params(self, learn_rate):

        """
        Updates the parameters of the neural network in the direction of reducing the loss.

        Args:
            learn_rate (float): Learning rate

        Returns:
            Nothing

        """

        layer = self.head

        while True:
            layer.update_params(learn_rate)

            if not layer.next:
                break

            layer = layer.next

    def training_step(self, x, y, learn_rate):

        """
        Perform a full training step using the backpropagation algorithm.

        Args:
            x (2d Numpy array): Training data x values.  Dimensions must be [batch_size x input_dim]
            y (2d Numpy array): Training data y values (i.e. labels).  Dimensions must be [batch_size x 1]
            learn_rate:

        Returns:
            Nothing

        """

        self.loss_layer.y = y

        self.forward_pass(x)
        self.backwards_pass()
        self.calc_gradients()
        self.update_params(learn_rate)
