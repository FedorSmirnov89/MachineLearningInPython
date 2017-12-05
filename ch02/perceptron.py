"""

Perceptron classifier class

"""

import numpy as np


class Perceptron(object):
    """
    Attributes:

        eta : float - learning rate (between 0.0 and 1.0)
        n_iter _ int - number of training iterations

    """

    def __init__(self, eta=.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
