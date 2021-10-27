"""Neural Network layers for crossbar array for the forward pass"""
from abc import ABC, abstractmethod

import numpy as np
from .. import utils
from ..lib_dpe_utils import xbar_mm
from ...nn import ops


class XbarLayer(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.Gmap = {}
        self.current = {}

    def __call__(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        return output

    @abstractmethod
    def _get_exp_param(self, *args, **kwargs):
        """
        get the target conductance map and input as well as there norm for the crossbar arrays
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Vector-matrix multiplication process in memory 'crossbar arrays'
        """
        raise NotImplementedError


#########################################################
#             Neural Network forward layers             #
#########################################################
class Linear(XbarLayer):
    """
    Linear layer for crossbar arrays
    """
    def __init__(self,
                 weight,
                 bias,
                 X,
                 w_g_ratio,
                 wm_type="dp_sparse",
                 normalize=False,
                 ):
        """
        :param weight: <np.ndarray> of shape (in_dim, out_dim)
                       weight parameters for linear layer
        :param bias: <np.ndarray> of shape (1, out_dim) or None
                     bias, if None, then no bias rows
        :param X: <np.ndarray> of shape (n_ex, in_dim)
                  input 2D volume
        :param w_g_ratio: int
        :param wm_type: dp_sparse, dp_dense, acm
                        weight mapping type
        :param normalize: bool
                          whether to normalize the weights after mapping
        """
        super().__init__()

        if bias is None:
            self.W = weight  # (in_dim, out_dim)
            self.X = X  # (n_ex, in_dim)
        else:
            self.W = np.append(weight, bias, axis=0)  # (in_dim + 1, out_dim)
            self.X = np.append(X, np.ones((X.shape[0], 1)), axis=1)  # (n_ex, in_dim + 1)

        self.w_g_ratio = w_g_ratio
        self.wm_type = wm_type
        self.normalize = normalize

    def _get_exp_param(self, **kwargs):
        """
        Get the target conductance map and input voltage vectors using 'Norm' and 'weight mapping' method
        """
        self.trgGmap, self.w_norm = utils.__dict__[self.wm_type](self.W,
                                                                 self.w_g_ratio,
                                                                 normalize=self.normalize,
                                                                 **kwargs)
        self.input, self.x_norm = utils.Norm(self.X.T, order=np.inf, axis=None)  # inf norm of input vectors

    def forward(self, dpe, array, position, **kwargs):
        """
        Forward pass of linear layer
        :param dpe: DPE class for superT platform
        :param array: n-tuple of int
        :param position: n-tuple of 2-tuple
        :return: output: <np.ndarray> of shape (n_ex, out_dim)
                         output of linear layer, shouldn't remove all the normalization operation for ourput
        """
        self._get_exp_param(**kwargs)
        Y, Gmap = xbar_mm(dpe, self.input, self.trgGmap, array, position, **kwargs)

        self.Gmap["Gmap"] = Gmap
        self.current["current"] = Y

        Y = Y[:, ::2] - Y[:, 1::2]

        output = Y * self.w_norm * self.x_norm.T / self.w_g_ratio

        return output


class conv2D(XbarLayer):
    """
    2D Convolutional layer for crossbar arrays
    """
    def __init__(self,
                 weight,
                 bias,
                 stride,
                 pad,
                 X,
                 w_g_ratio,
                 wm_type="dp_sparse",
                 normalize=False,
                 ):
        """
        :param weight: <np.ndarray> of shape (out_ch, in_ch, fr, fc)
                       2D convolutional layer weights
        :param bias: <np.ndarray> of shape (out_ch, 1)
                     bias term
        :param stride: int
        :param pad: tuple, int, 'same'
        :param X: <np.ndarray> of shape (n_ex, in_ch, in_rows, in_cols)
        :param w_g_ratio: int
        :param wm_type: dp_sparse, dp_dense, acm
                        weight mapping type
        :param normalize: bool
                          whether to normalize the weights after mapping
        """

        super().__init__()

        s = stride
        self.out_ch, in_ch, fr, fc = weight.shape
        self.n_ex, in_ch, in_rows, in_cols = X.shape

        _, p = ops.pad2D(X, pad, (fr, fc), s)
        pr1, pr2, pc1, pc2 = p

        self.out_rows = (in_rows + pr1 + pr2 - fr) // s + 1
        self.out_cols = (in_cols + pc1 + pc2 - fc) // s + 1

        if bias is None:
            self.W = weight.reshape(self.out_ch, -1).T  # (fr * fc * in_ch, out_ch)
            self.X = ops.im2col(X, weight.shape, pad, stride)  # (fr * fc * in_ch, n_ex * out_rows * out_cols)
        else:
            # (fr * fc * in_ch + 1, out_ch)
            self.W = np.append(weight.reshape(self.out_ch, -1), bias, axis=1).T
            # (fr * fc * in_ch + 1, n_ex * out_rows * out_cols)
            self.X = np.insert(ops.im2col(X, weight.shape, pad, stride), fr * fc * in_ch, 1, axis=0)

        self.w_g_ratio = w_g_ratio
        self.wm_type = wm_type
        self.normalize = normalize

    def _get_exp_param(self, **kwargs):
        """
        Get the target conductance map and input voltage vectors using 'Norm' and 'weight mapping' method
        """
        self.trgGmap, self.w_norm = utils.__dict__[self.wm_type](self.W,
                                                                 self.w_g_ratio,
                                                                 normalize=self.normalize,
                                                                 **kwargs)
        self.input, self.x_norm = utils.Norm(self.X, order=np.inf, axis=None)  # inf norm of input vectors

    def forward(self, dpe, array, position, **kwargs):
        """
        Forward pass of linear layer
        :param dpe: DPE class for superT platform
        :param array: n-tuple of int
        :param position: n-tuple of 2-tuple
        :return: output: <np.ndarray> of shape (n_ex, out_dim)
                         output of linear layer, shouldn't remove all the normalization operation for ourput
        """
        self._get_exp_param(**kwargs)
        Y, Gmap = xbar_mm(dpe, self.input, self.trgGmap, array, position, **kwargs)

        self.Gmap["Gmap"] = Gmap
        self.current["current"] = Y

        Y = Y[:, ::2] - Y[:, 1::2]

        output = Y * self.w_norm * self.x_norm.T / self.w_g_ratio  # (n_ex * out_rows * out_cols, out_ch)
        output = output.reshape(self.out_rows, self.out_cols, self.n_ex, self.out_ch).transpose(2, 3, 0, 1)

        return output









