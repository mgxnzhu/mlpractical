import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import argparse
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit, GlorotNormalInit, SELUInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')
parser.add_argument('--layer_type', nargs="?", type=str, default='Relu',
    help='Your activation (LeakyRelu, ELU, SELU) used in the test')
parser.add_argument('--layer_num', nargs="?", type=int, default=2,
    help='Number of hidden layers')
parser.add_argument('--init_type', nargs="?", type=str, default='Uniform',
    help='"Uniform" or "Normal"')
parser.add_argument('--init_mode', nargs="?", type=str, default='inout',
    help='"in", "out", or "inout"')

args = parser.parse_args()
layer_type = args.layer_type
layer_num = args.layer_num
init_type = args.init_type
init_mode = args.init_mode

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

    np.save(layer_type+'_'+str(layer_num)+'_'+init_type+'_'+init_mode+'_train_stat.npy',stats)
    #np.save('SELUInit.npy',stats)
    return stats, keys, run_time
'''
    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')
    #fig_1.savefig(layer_type+'_'+str(layer_num)+'_error.png')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    #fig_2.savefig(layer_type+'_'+str(layer_num)+'_accuracy.png')
''' 
   
#    return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2


# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=batch_size, rng=rng)

# The model set up code below is provided as a starting point.
# You will probably want to add further code cells for the
# different experiments you run.

#setup hyperparameters
learning_rate = 0.01
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 10, 100

weights_init = eval('Glorot'+init_type+'Init(mode="%s",rng=rng)'%init_mode)

# ---- Special Set for SELUInit ----
#layer_num = 4
#layer_type = 'SELU'
#weights_init = SELUInit(rng=rng)
# ----

biases_init = ConstantInit(0.)

input_layer = AffineLayer(input_dim, hidden_dim, weights_init, biases_init)
hidden_layer = AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init)
output_layer = AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
act_layer = eval(layer_type+'Layer()')

model = MultipleLayerModel([input_layer, act_layer]+[hidden_layer, act_layer]*layer_num+[output_layer])

error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
_ = train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False)
