# Deep-Learning-with-Numpy

A very simple deep learning library implemented in Numpy.

```python
network = NeuralNetwork()

# Build network with 6 - 4 - 1 units
network.add_input_layer(x_train.shape[1], n_out=6)
network.add_sigmoid_activation()
network.add_linear_layer(n_out=4)
network.add_sigmoid_activation()
network.add_output_layer(n_out=1)
network.add_mse_loss_layer()

for _ in range(steps):
    network.training_step(x_train, y_train, learn_rate=0.01)
```

Even though it is at at totally different scale, putting this together has 
given me a feeling for how frameworks such as Keras and Tensorflow have come
 together, and the kind of design decisions that go into these tools. 

I also used it as an opportunity to use [sphinx](http://www.sphinx-doc.org/en/master/), [pytest](https://docs.pytest.org/en/latest/) and the 
linters [pylint](https://www.pylint.org/), [pycodestyle](https://pypi.org/project/pycodestyle/) and [pydocstyle](https://github.com/PyCQA/pydocstyle), 
which were all new to me.

The packages I am using are in `requirements.txt` but if all you want to do is run the code then all you should need is Numpy.

To get set up and running this code in a conda virtual environment, run the following in the command line:

```shell
git clone https://github.com/jonathan-smith-1/dl_with_numpy
cd dl_with_numpy
conda create -n my_env_name python
source activate my_env_name
pip install .
python run.py
```
