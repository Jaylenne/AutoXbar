"""A collection of composable layer object for building neural network"""
from abc import ABC, abstractmethod

import numpy as np


class FunctionBase(ABC):
    """
    An abstract base class for all differentiable functions in neural networks
    """
    def __init__(self, *args, **kwargs):
        # cache for intermediate result
        self.cache = {}
        # whether the function need to be trained
        self.trainable = True
        # local gradients of output wrt. input for backward propagation
        self.grad = {}

    def __call__(self, *args, **kwargs):
        # calculate the forward pass
        output = self.forward(*args, **kwargs)
        # register local gradients into self.grad
        self.grad= self.local_grad(*args, **kwargs)

        return output

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the function. Calculate the output value
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwargs):
        """
        Backward pass of the function. Calculate the global gradients of loss wrt. input value
        """
        raise NotImplementedError

    def local_grad(self, *args, **kwargs):
        """
        Calculates the local gradients of the output value wrt. input value
        """
        raise NotImplementedError


class LayerBase(FunctionBase, ABC):
    """
    An abstract class for neural network layers in neural networks. This includes the parameters of weights
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = {}
        self.weight_update = {}

    @abstractmethod
    def _init_weights(self, *args, **kwargs):
        """
        Weight initialization for parameters. Each layer has a default weight initialization. Each layer can also be
        initialized by weight initialization method.
        """
        raise NotImplementedError



