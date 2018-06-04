class NeuralNetwork():

    def __init__(self):

        """
        nodes: List of integers.  Number of nodes in each layer. including the final layer.
        """

        # Computation graph is a doubly-linked list
        self.head = None
        self.tail = None

        # Record of which layers are output and loss
        self.output_layer = None
        self.loss_layer = None

    def add_hidden_layer(self, new_layer):

        # Add new layer to doubly-linked list computation graph
        if self.tail:
            self.tail.next = new_layer
            new_layer.prev = self.tail
        else:
            self.head = self.tail = new_layer

        self.tail = new_layer

    def add_output_layer(self, new_layer):

        self.add_hidden_layer(new_layer)
        self.output_layer = new_layer

    def add_loss_layer(self, new_layer):

        self.add_hidden_layer(new_layer)
        self.loss_layer = new_layer

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