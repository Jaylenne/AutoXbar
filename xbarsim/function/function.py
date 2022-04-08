import numpy as np
import torch

from ..xbar import xbar, add_program_noise
from ...xbar import utils
from ...nn import ops


def linear(Xbar, A, B, **kwargs):
    """
    Matrix multiplication in crossbar arrays
    A multiplies B
    """
    normalize = kwargs.get('normalize', False)
    bias = kwargs.get('bias', False)
    if bias is not False:
        if bias.ndim == 1:
            bias = bias[None, :]
        B = np.append(B, bias, axis=0)
        A = np.append(A, np.ones((A.shape[0], 1)), axis=1)
    G, G_norm = utils.__dict__[Xbar.wm_type](B,
                                             Xbar.w_g_ratio,
                                             normalize=normalize)
    vvec, v_norm = utils.Norm(A, order=np.inf, axis=None)
    vvec = vvec * Xbar.Vread

    current = Xbar.mm(vvec, G)
    if Xbar.wm_type == 'ACM':
        output = current[:, :-1] - current[:, 1:]
    else:
        output = current[:, ::2] - current[:, 1::2]
    output = (output * G_norm * v_norm.T) / (Xbar.w_g_ratio * Xbar.Vread)

    return output


def Meff(Xbar, A, **kwargs):
    """
    Get effective storage of the matrix by adding noise and wire resistance effect
    """
    normalize = kwargs.get('normalize', False)
    G, G_norm = utils.__dict__[Xbar.wm_type](A,
                                             Xbar.w_g_ratio,
                                             normalize=normalize)
    G = Xbar._cal_gmap(G)
    Geff = G * G_norm / Xbar.w_g_ratio
    if Xbar.wm_type == 'ACM':
        Aeff = Geff[:, :-1] - Geff[:, 1:]
    else:
        Aeff = Geff[:, ::2] - Geff[:, 1::2]
    return Aeff


def conv2D(Xbar, A, B, stride, pad, **kwargs):
    """
    2D convolution by crossbar

    Parameters
    ----------
    Xbar: xbar class with attributions
    A: Matrix A to feed into crossbar as voltage inputs, shape [n_ex, in_ch, in_rows, in_cols]
    B: Matrix B to be stored in the crossbar as conductance, shape [out_ch, in_ch, fr, fc]
    stride: int
    pad: tuple, int, or 'same'
    kwargs: bias, [out_ch, 1] or [out_ch, ]
            normalize, True or False, whether to normalize the input matrix

    Returns
    -------
    Output value: not current but real value [n_ex, out_ch, out_rows, out_cols]
    """
    normalize = kwargs.get('normalize', False)
    bias = kwargs.get('bias', False)

    s = stride
    out_ch, in_ch, fr, fc = B.shape
    n_ex, in_ch, in_rows, in_cols = A.shape

    _, p = ops.pad2D(A, pad, (fr, fc), s)
    pr1, pr2, pc1, pc2 = p

    out_rows = (in_rows + pr1 + pr2 - fr) // s + 1
    out_cols = (in_cols + pc1 + pc2 - fc) // s + 1

    if not bias:
        W = B.reshape(out_ch, -1).T  # [fr * fc * in_ch, out_ch]
        X = ops.im2col(A, B.shape, pad, stride).T  # (n_ex * out_rows * out_cols, fr * fc * in_ch)

    if bias is not False:
        if bias.ndim == 1:
            bias = bias[:, None]  # [out_ch, 1]
            W = np.append(B.reshape(out_ch, -1), bias, axis=1).T  # [fr * fc * in_ch + 1, out_ch]
            # (n_ex * out_rows * out_cols, fr * fc * in_ch + 1)
            X = np.insert(ops.im2col(A, B.shape, pad, stride), fr * fc * in_ch, 1, axis=0)

    # Get input and normalized Conductance Matrix
    G, G_norm = utils.__dict__[Xbar.wm_type](W,
                                             Xbar.w_g_ratio,
                                             normalize=normalize)
    vvec, v_norm = utils.Norm(X, order=np.inf, axis=None)
    vvec = vvec * Xbar.Vread

    # Matrix Mulyiplication
    current = Xbar.mm(vvec, G)
    if Xbar.wm_type == 'ACM':
        output = current[:, :-1] - current[:, 1:]
    else:
        output = current[:, ::2] - current[:, 1::2]
    output = (output * G_norm * v_norm.T) / (Xbar.w_g_ratio * Xbar.Vread)

    # Reshape the output into desired form
    output = output.reshape(out_rows, out_cols, n_ex, out_ch).transpose(2, 3, 0, 1)
    return output
