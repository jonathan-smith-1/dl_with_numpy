import abc


class Layer(metaclass=abc.ABCMeta):
    """
    Base class for a single layer of neural network.  'Layer' includes activation and loss layers.
    """

    def __init__(self, n_in, n_out):

        """
        Initialise attributes of base class.

        Args:
            n_in (integer): Size of input to this layer.  This layer accepts inputs with dimension [batch_size, n_in].
            n_out (integer): Size of output of this layer.  This layer creates outputs with dimension
                             [batch_size, n_out]
        """

        self.input = None
        self.output = None

        self.input_size = n_in
        self.output_size = n_out

        self.dL_din = None  # derivative of loss w.r.t. input
        self.dL_dout = None  # derivative of loss w.r.t. output

        self.next = None  # next node in the computation graph
        self.prev = None  # previous node in the computation graph

    @abc.abstractmethod
    def forward_pass(self):

        """
        Calculate the output of this layer from its input and store the result.

        Returns:
            Nothing
        """

    @abc.abstractmethod
    def backward_pass(self):

        """
        Calculate the derivative of this layer's input with respect to the loss from the derivative of this layer's
        output with respect to the loss.  Store the result.

        Returns:
            Nothing
        """

    def calc_param_grads(self):

        """
        Calculate the gradient of the loss with respect to this layer's parameters, if there are any.  Store the result.

        Returns:
            Nothing
        """

    def update_params(self, lr):

        """
        Update this layer's parameters, if there are any.

        Args:
            lr (float): Learning rate

        Returns:
            Nothing

        """


