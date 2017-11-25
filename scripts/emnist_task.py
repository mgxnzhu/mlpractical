import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider

from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.layers import ConvolutionalLayer, MaxPoolingLayer, ReshapeLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, RMSPropLearningRule, AdamLearningRule
from mlp.optimisers import Optimiser


def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    np.save('SimpleCNN_stats.npy',stats)
    #np.save('SimpleCNN_keys.npy',keys)
    #np.save('SimpleCNN_runtime.npy',run_time)
    return stats, keys, run_time

def emnist_task():

    # Seed a random number generator
    seed = 10102016 
    rng = np.random.RandomState(seed)
    batch_size = 100
    # Set up a logger object to print info about the training run to stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [logging.StreamHandler()]

    # Create data provider objects for the MNIST data set
    train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
    valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)

    learning_rate = 0.0001
    num_epochs = 5
    stats_interval = 1

    conv_kernel_dim1, conv_kernel_dim2 = 5, 5
    pool_kernel_dim1, pool_kernel_dim2 = 2, 2
    conv1_channel, conv1_dim1, conv1_dim2 = 5, 28, 28
    pool1_dim1, pool1_dim2 = 24, 24
    conv2_channel, conv2_dim2, conv2_dim2 = 10, 12, 12
    pool2_dim1, pool2_dim2 = 8, 8
    fullcon_dim, output_dim, hidden_dim = 160, 47, 100

    weights_init = GlorotUniformInit(rng=rng)
    biases_init = ConstantInit(0.)
    model = MultipleLayerModel([
        ReshapeLayer((1, conv1_dim1, conv1_dim2)),
        ConvolutionalLayer(1, conv1_channel, conv1_dim1, conv1_dim2, conv_kernel_dim1, conv_kernel_dim2),
        MaxPoolingLayer(conv1_channel, pool1_dim1, pool1_dim2, pool_kernel_dim1, pool_kernel_dim2),
        ReluLayer(),
        ConvolutionalLayer(conv1_channel, conv2_channel, conv2_dim2, conv2_dim2, conv_kernel_dim1, conv_kernel_dim2),
        MaxPoolingLayer(conv2_channel, pool2_dim1, pool2_dim2, pool_kernel_dim1, pool_kernel_dim2),
        ReshapeLayer(),
        AffineLayer(fullcon_dim, hidden_dim, weights_init, biases_init),
        ReluLayer(),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init),
        SoftmaxLayer()
    ])

    error = CrossEntropySoftmaxError()
    # Use a basic gradient descent learning rule
    learning_rule = AdamLearningRule(learning_rate=learning_rate)

    #Remember to use notebook=False when you write a script to be run in a terminal
    _ = train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False)

emnist_task()
