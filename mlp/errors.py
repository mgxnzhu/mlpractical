# -*- coding: utf-8 -*-
"""Error functions.

This module defines error functions, with the aim of model training being to
minimise the error function given a set of inputs and target outputs.

The error functions will typically measure some concept of distance between the
model outputs and target outputs, averaged over all data points in the data set
or batch.
"""

import numpy as np


class SumOfSquaredDiffsError(object):
    """Sum of squared differences (squared Euclidean distance) error."""

    def __call__(self, outputs, targets):
        """Calculates error function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar error function value.
        """
        N = outputs.shape[0]
        error = np.sum((outputs-targets)**2)/(2*N)
        return error
        #raise NotImplementedError()

    def grad(self, outputs, targets):
        """Calculates gradient of error function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of error function with respect to outputs. This should be
            an array of shape (batch_size, output_dim).
        """
        N = outputs.shape[0]
        error_grad = (outputs-targets)/N
        return error_grad
        #raise NotImplementedError()

    def __repr__(self):
        return 'SumOfSquaredDiffsError'
