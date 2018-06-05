# Deep-Learning-with-Numpy

A very simple deep learning library implemented in Numpy.

```python
network = NeuralNetwork()

network.add_input_layer(x_train.shape[1], n_out=6)
network.add_sigmoid_activation()
network.add_linear_layer(n_out=4)
network.add_sigmoid_activation()
network.add_output_layer(n_out=1)
network.add_loss_layer()

for _ in range(iterations):
    network.training_step(x_train, y_train, learn_rate=0.01)
```

Even thought it is at at totally different scale, putting this together has given me a feeling for how frameworks such as Keras and Tensorflow have come together. 

I also used it as an opportunity to use *sphinx* and *pytest*, which were both new to me.

The packages I am using are in `requirements.txt` but if all you want to do is run the code then all you should need is Numpy.

To update documentation, run the following command from within the docs directory:

```shell
cd docs
make clean
make html
```

To run the unit tests, in the `dl_with_numpy` directory, run:

```shell
cd docs
py.test
```
