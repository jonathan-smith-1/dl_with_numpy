# Deep-Learning-with-Numpy

A very simple deep learning library implemented in Numpy.

```python
network = NeuralNetwork()

network.add_input_layer(x_train.shape[1], n_out=6)
network.add_sigmoid_activation()
network.add_linear_layer(n_out=4)
network.add_sigmoid_activation()
network.add_output_layer(n_out=1)
network.add_mse_loss_layer()

for _ in range(iterations):
    network.training_step(x_train, y_train, learn_rate=0.01)
```

Even thought it is at at totally different scale, putting this together has given me a feeling for how frameworks such as Keras and Tensorflow have come together. 

I also used it as an opportunity to use *sphinx* and *pytest*, which were both new to me.

The packages I am using are in `requirements.txt` but if all you want to do is run the code then all you should need is Numpy.

To get set up and running this code in a conda virtual environment, run the following in the command line:

```shell
git clone https://github.com/jonathan-smith-1/dl_with_numpy
cd dl_with_numpy
conda create -n my_env_name python
source activate my_env_name
pip install -r requirements.txt
python run.py
```
If you change any docstrings, to update the documentation run the following command from within the `docs` directory:

```shell
make clean
make html
```

To run the unit tests, in the root `dl_with_numpy` directory, run from the command line:

```shell
py.test
```
