"""Example use of Popsearch.

Tunes hyper-parameters of a feed-forward network to to predict
:math:`y(x) = 1 + 0.3 * x_1 - 0.6 * x_2^2 - 0.2 * x_3^3 + 0.5 x_4^4`.

Hyper-parameters that we tune:

    - Param init scale
    - Number of layers
    - Hidden layer size
    - Activation function
    - Learning rate

To run this example, you need numpy and the autograd package
(https://github.com/HIPS/autograd/), a lightweight autodiff
library for numpy. Run the example with

    >>> python main.py
"""
import numpy as np
import autograd.numpy as auto_np
from autograd import grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam, sgd

from popsearch import run, Parameter, Config


#### Toy data

def build_data(seed):
    """Build toy data set"""
    rs = np.random.RandomState(seed)

    def y(x):
        """ y(x) = 1 + 0.3 * x_1 - 0.6 * x_2^2 - 0.2 * x_3^3 + 0.5 x_4^4 """
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return 1 + 0.3 * x1 - 0.6 * x2 ** 2 - 0.2 * x3 ** 3 + 0.5 * x4 ** 4

    xtrain = rs.rand(10000, 4)
    xtest = rs.rand(1000, 4)
    ytrain = y(xtrain) + rs.rand(10000) / 10
    ytest = y(xtest) + rs.rand(1000) / 10
    return xtrain, xtest, ytrain, ytest

#### Model
# from https://github.com/HIPS/autograd/blob/master/examples/neural_net_regression.py


def sigmoid(x):
    return 1 / (1 + auto_np.exp(-x))


def relu(x):
    return (x > 0) * x


def init_random_params(scale, layer_sizes, seed=0):
    """Build a list of (weights, biases) tuples, one for each layer."""
    rs = npr.RandomState(seed)
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


def nn_predict(params, inputs, nonlinearity=auto_np.tanh):
    """Forward pass of network"""
    for W, b in params:
        outputs = auto_np.dot(inputs, W) + b
        inputs = nonlinearity(outputs)
    return outputs


def mse(weights, inputs, targets, nonlinearity=auto_np.tanh):
    """Negative log-likelihood objective"""
    predictions = nn_predict(weights, inputs, nonlinearity)
    return auto_np.mean((targets - predictions) ** 2)


def pop_train(state):
    """The popsearch objective"""
    ival = state.parameters['ival']
    N = state.config.n_step * ival
    bsz = state.parameters['bsz']
    seed = state.parameters['seed']
    rs = np.random.RandomState(seed)
    nhid = state.parameters['nhid']
    nlayers = state.parameters['nlayers']
    init_size = state.parameters['init_size']
    lr = state.parameters['lr']
    optim = {'sgd': sgd, 'adam': adam}[state.parameters['optim']]
    activation = {'sig': sigmoid, 'relu': relu, 'tanh': auto_np.tanh}[
        state.parameters['activation']]

    sizes = [4] + [nhid] * (nlayers - 1) + [1]
    params = init_random_params(init_size, sizes, seed)

    xtrain, xtest, ytrain, ytest = build_data(seed)

    def objective(weights, t):
        idx = rs.permutation(xtrain.shape[0])[:bsz]
        batch_in = xtrain[idx]
        batch_ta = xtrain[idx]
        return mse(weights, batch_in, batch_ta, nonlinearity=activation)

    def callback(weights, i, grad):
        if i % ival == 0:
            state.eval(mse(weights, xtest, ytest, nonlinearity=activation))
        return

    optim(grad(objective), params, step_size=lr, num_iters=N, callback=callback)
    return


if __name__ == '__main__':
    import os
    logs = os.listdir('./log')
    for log in logs:
        if log.endswith('.log'):
            log = './log/' + log
            os.remove(log)
        else:
            fig = log
            figs = os.listdir('./log/' + fig)
            for fig in figs:
                fig = './log/figs/' + fig
                os.remove(fig)

    params = [
        Parameter('bsz', int, frozen=20),
        Parameter('seed', int, frozen=0),
        Parameter('nhid', int, support=list(range(2, 100))),
        Parameter('nlayers', int, support=list(range(1, 3))),
        Parameter('lr', float, minmax=(0.0001, 0.01)),
        Parameter('init_size', float, minmax=(0.1, 1)),
        Parameter('activation', str, support=['sig', 'tanh', 'relu']),
        Parameter('optim', str, support=['sgd', 'adam']),
        Parameter('ival', int, frozen=1),
    ]

    config = Config(
        callable=pop_train,
        path='./log',
        n_step=100,
        n_pop=10,
        n_job=4,
        buffer=2,
        max_val=1,
        sleep=0.1,
        p_force=0,
        perturb=(0.1, 1.0, 0.95, 0.01),
        eval_rule=('double', None, ((1.,)), 2),
        perturb_rule=('sample', 'pareto', ((1.5,))),
        plot=True,
        plot_config={'save': False, 'semilogy': False},
        seed=133,
    )

    run(config, params)
