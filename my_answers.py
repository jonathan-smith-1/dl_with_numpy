import numpy as np

class Layer():

    """
    Base class for a single layer of neural network.
    """

    def __init__(self):
        self.input = None
        self.output = None
        self.dL_din = None    # derivative of loss w.r.t. input
        self.dL_dout = None   # derivative of loss w.r.t. output

        self.next = None    # next node in the computation graph
        self.prev = None    # previous node in the computation graph

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

        super(LinearLayer, self).__init__()

        np.random.seed(seed)
        self.W = np.random.normal(0.0, n_in**-0.5, (n_in, n_out))  # parameters of the linear layer
        self.dL_dW = None  # derivative of loss w.r.t. parameters W

    def forward_pass(self):
        self.output = np.dot(self.input, self.W)

    def backward_pass(self):
        self.dL_din = np.dot(self.dL_dout, self.W.T)

    def calc_param_grads(self):
        self.dL_dW = np.dot(self.input.T, self.dL_dout)

    def update_params(self, lr):
        self.W -= self.dL_dW * lr


class SigmoidActivation(Layer):

    """
    Sigmoid activation layer of neural network

    n: Integer. Size of layer
    """

    def __init__(self, n):

        super(SigmoidActivation, self).__init__()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def forward_pass(self):
        self.output = self.sigmoid(self.input)

    def backward_pass(self):
        self.dL_din = self.dL_dout * self.sigmoid_deriv(self.input)


class MeanSquareLoss(Layer):

    """
    Mean square loss function

    """

    def __init__(self):

        super(MeanSquareLoss, self).__init__()

        self.dL_dout = 1  # output of this layer is the loss, so this derivative is 1
        self.mean_sq_loss = None
        self.y = None   # true output

    def forward_pass(self):
        self.mean_sq_loss = 0.5*np.mean(np.square(self.input - self.y))

    def backward_pass(self):
        self.dL_din = (self.input - self.y)/self.y.size


class NeuralNetwork():

    def __init__(self, num_nodes):

        """
        nodes: List of integers.  Number of nodes in each layer. including the final layer.
        """

        self.layers = self.construct_layers(num_nodes)   # list of the layers objects, from input to output

        self.link_layers(self.layers)   # link the layers together

        self.first_layer = self.layers[0]
        self.output_layer = self.layers[-2]
        self.loss_layer = self.layers[-1]


    def construct_layers(self, num_nodes):

        layers = []
        in_idx = 0
        out_idx = 1

        while out_idx < len(num_nodes) - 1:

            layers.append(LinearLayer(num_nodes[in_idx], num_nodes[out_idx]))   # add the linear layer
            layers.append(SigmoidActivation(num_nodes[out_idx]))           # add the activation layer

            in_idx += 1
            out_idx += 1

        layers.append(LinearLayer(num_nodes[-2], num_nodes[-1]))    # add the final linear layer

        layers.append(MeanSquareLoss())

        return layers

    def link_layers(self, layers):

        """
        Populate the self.next and self.prev attributes of each Layer object to link them together
        """

        for i in range(len(layers) - 1):
            layers[i].next = layers[i+1]

        for i in range(1, len(self.layers)):
            layers[i].prev = layers[i-1]

    def forward_pass(self, x, y):

        """
        Performs forward pass through neural network

        x: 2d numpy array.  Input to first layer.
        y: 2d numpy array.  True labels.
        """

        self.first_layer.input = x
        self.loss_layer.y = y

        layer = self.first_layer
        while layer.next:
            layer.forward_pass()
            layer.next.input = layer.output   # propagate forward pass
            layer = layer.next

        # forward pass on final layer
        layer.forward_pass()

    def backwards_pass(self, y):

        """
        Performs the backwards pass of the neural network.

        y: 2D numpy array of the output.
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

        layer = self.first_layer

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

        layer = self.first_layer

        while True:
            layer.update_params(learn_rate)

            if not layer.next:
                break

            layer = layer.next

    def training_step(self, x, y, learn_rate):

        self.forward_pass(x, y)
        self.backwards_pass(y)
        self.calc_gradients()
        self.update_params(learn_rate)


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 2000
learning_rate = 1.5
hidden_nodes = 6
output_nodes = 1
