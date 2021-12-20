import numpy as np
import torch

from ..xbar import xbar, add_program_noise
from ...xbar import utils


def linear(Xbar, A, B, **kwargs):
    """
    Matrix multiplication in crossbar arrays
    A multiplies B
    """
    normalize = kwargs.get('normalize', True)
    G, G_norm = utils.__dict__[Xbar.wm_type](B,
                                             Xbar.w_g_ratio,
                                             normalize=normalize,
                                             **kwargs)
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
    Get effective storage of the matrix
    """
    normalize = kwargs.get('normalize', False)
    G, G_norm = utils.__dict__[Xbar.wm_type](A,
                                             Xbar.w_g_ratio,
                                             normalize=normalize,
                                             **kwargs)
    G = add_program_noise(G)
    Geff = G * G_norm / Xbar.w_g_ratio
    if Xbar.wm_type == 'ACM':
        Aeff = Geff[:, :-1] - Geff[:, 1:]
    else:
        Aeff = Geff[:, ::2] - Geff[:, 1::2]
    return Aeff
