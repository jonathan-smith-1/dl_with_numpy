.. dl_with_numpy documentation master file, created by
   sphinx-quickstart on Mon Jun 18 15:22:13 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation for dl_with_numpy
==============================================
Build and train simple neural networks, for example::

   x_train = np.array([[1., 2., 3.],
                       [8., 6., 3]])

   y_train = np.array([[15.],
                       [41.]])

   network = NeuralNetwork()

   # Build network with 6 - 4 - 1 units
   network.add_input_layer(x_train.shape[1], n_out=6)
   network.add_sigmoid_activation()
   network.add_linear_layer(n_out=4)
   network.add_sigmoid_activation()
   network.add_output_layer(n_out=1)
   network.add_mse_loss_layer()

   steps = 250
   for _ in range(steps):
       network.training_step(x_train, y_train, learn_rate=0.01)


And test them::

    x_test = np.array([[1., 2., 3.]])
    network.forward_pass(x_test)
    prediction = network.output_layer.output

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
