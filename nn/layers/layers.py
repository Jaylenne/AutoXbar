"""A collection of composable layer object for building neural network"""
from abc import ABC, abstractmethod

import numpy as np


class FunctionBase(ABC):
    """
    An abstract base class for all differentiable functions in neural networks
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
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
        self.grad = self.local_grad(*args, **kwargs)

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

    @abstractmethod
    def local_grad(self, *args, **kwargs):
        """
        Calculates the local gradients of the output value wrt. input value
        """
        raise NotImplementedError


class LayerBase(FunctionBase):
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


#########################################################
#                  Function Layers                      #
#########################################################

class Add(FunctionBase):
    def forward(self, X):
        """
        :param X: list of <np.ndarray> of shape (n_ex, *)
                  list of tuple of input tensors with the same size
        :return:  <np.ndarray> of shape (n_ex, *)
                  output of add operation
        """
        output = X[0].copy()
        for i in range(1, len(X)):
            output += X[i]

        self.cache["X"] = X
        self.cache["output"] = output

        return output

    def backward(self, dLdY):
        """
        :param dLdY: <np.ndarray> of shape (n_ex, *)
                     global gradients wrt. output value
        :return: list of <np.ndarray> of shape (n_ex, *)
                 list of global gradients wrt. input value
        """
        grads = [dLdY * dx for dx in self.grad["X"]]
        return grads

    def local_grad(self, X):
        """
        :param X: list of <np.ndarray> of shape (n_ex, *)
                  list of tuple of input tensors with the same size
        :return: list of local gradients [1] * length of X
                 local gradients of output value wrt. input
        """
        dX = [1.] * len(X)
        grads = {"X": dX}
        return grads


class Flatten(FunctionBase):
    def forward(self, X):
        """
        :param X: <np.ndarray>
                  tensors to be flatten to 1 dimension
        :return: <np.ndarray> of shape (n_ex, -1)
                 flattened tensors
        """
        self.cache["shape"] = X.shape
        n_batch = X.shape[0]

        return X.reshape(n_batch, -1)

    def backward(self, dLdY):
        """
        :param dLdY: <np.ndarray> of shape (n_ex, -1)
                     global gradients of loss wrt. flattened tensors
        :return: <np.ndarray>
                 global gradients of loss wrt. input tensor
        """
        return (dLdY * self.grad["X"]).reshape(self.cache["shape"])

    def local_grad(self, X):
        dX = 1.
        grads = {"X": dX}
        return grads



#########################################################
#                  Neural Network Layers                #
#########################################################
class Linear()







