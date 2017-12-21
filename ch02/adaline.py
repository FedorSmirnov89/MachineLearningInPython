"""
File containing the adaline neuron classifier

"""

import numpy as np

class AdalineGD(object):
    """
    Parameters:

        eta: float (0.0 - 1.0) learning rate
        n_iter: int number of iterations for the training data


    Attributes:

        w_ : 1d-array The weights of the neuron
        errors_ : list Number of false classifications per epoch
    """

    def __init__(self, eta=.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return  self