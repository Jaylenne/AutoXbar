"""DPE utils for controlling writing, reading and vector-matrix-multiplication (VMM)"""
import numpy as np

from ..utils import Xbar_tile_aggre


def Vec_pn(vec):
    """
    Split vec into positive and negative parts
    :param vec: vector to be multiplied
    :return: vec_pos, vec_neg: positive and negative part of input vectors
    """
    if np.ndim(vec) == 1:
        vec = np.expand_dims(vec, 1)

    vec_pos, vec_neg = vec.copy(), vec.copy()
    vec_pos[vec_pos < 0] = 0
    vec_neg[vec_neg > 0] = 0

    return vec_pos, -vec_neg


def dpe_pn(dpe, Vec, array, c_sel, **kwargs):
    """
    Crossbar VMM for vectors including negative value
    :param dpe: DPE class for superT platform
    :param Vec: <np.ndarray> of shape (64, *)
                vector to be multiplied
    :param array: 0, 1, 2
    :param c_sel: [c_start, c_end]
    :param kwargs: other parameters
    :return:
    """
    Vread = kwargs['Vread'] if 'Vread' in kwargs.keys() else 0.2
    tdly = kwargs['tdly'] if 'tdly' in kwargs.keys() else 1000

    Vec_pos, Vec_neg = Vec_pn(Vec)

    Ipos = dpe.multiply(array,
                        Vec_pos,
                        c_sel=c_sel,
                        r_start=0, mode=0, Tdly=tdly)

    Ineg = dpe.multiply(array,
                        Vec_neg,
                        c_sel=c_sel,
                        r_start=0, mode=0, Tdly=tdly)

    Ires = Ipos - Ineg

    return Ires


def vec_mm(dpe, array, vec, geff, start_pos, **kwargs):
    """
    program conductance and do the matrix multiplications
    :param dpe: DPE class for superT platform
    :param array: 0, 1, 2
    :param vec: <np.ndarray> of 2D
                vectors to be multiplied
    :param geff: <np.ndarray> of 2D
                 target conductance map
    :param start_pos: 2-tuple
                      the position that geff stays in the 64x64 array
    :param lincor: bool
                   whether doing linear correction
    :return output: <np.ndarray>
            gmap: the according conductance map of geff
    """
    vSetRamp = kwargs['vSetRamp'] if 'vSetRamp' in kwargs.keys() else [1.0, 2.5, 0.1]
    vGateSetRamp = kwargs['vGateSetRamp'] if 'vGateSetRamp' in kwargs.keys() else [0.95, 2.0, 0.05]
    vResetRamp = kwargs['vResetRamp'] if 'vResetRamp' in kwargs.keys() else [0.5, 3.5, 0.05]
    vGateResetRamp = kwargs['vGateResetRamp'] if 'vGateResetRamp' in kwargs.keys() else [5.0, 5.5, 0.1]

    maxSteps = kwargs['maxSteps'] if 'maxSteps' in kwargs.keys() else 10

    Gtol = kwargs['Gtol'] if 'Gtol' in kwargs.keys() else 5e-6
    Gtol_in = kwargs['Gtol_in'] if 'Gtol_in' in kwargs.keys() else Gtol
    Gtol_out = kwargs['Gtol_out'] if 'Gtol_out' in kwargs.keys() else Gtol

    # Msel = kwargs['Msel'] if 'Msel' in kwargs.keys() else np.ones(self.shape)

    saveHistory = kwargs['saveHistory'] if 'saveHistory' in kwargs.keys() else False
    maxRetry = kwargs['maxRetry'] if 'maxRetry' in kwargs.keys() else 5

    Tdly = kwargs['Tdly'] if 'Tdly' in kwargs.keys() else 1000
    method = kwargs['method'] if 'method' in kwargs.keys() else 'slow'

    Twidth = kwargs['Twidth'] if 'Twidth' in kwargs.keys() else 1000e-6
    TwidthSet = kwargs['TwidthSet'] if 'TwidthSet' in kwargs.keys() else Twidth
    TwidthReset = kwargs['TwidthReset'] if 'TwidthReset' in kwargs.keys() else Twidth

    Vread = kwargs['Vread'] if 'Vread' in kwargs.keys() else 0.2
    tdly = kwargs['tdly'] if 'tdly' in kwargs.keys() else 500

    lincor = kwargs['lincor'] if 'lincor' in kwargs.keys() else True

    assert vec.shape[0] == geff.shape[0], "dimension doesn't match!"
    assert geff.shape[0] + start_pos[0] <= 64, "mapping exceed number of rows!"
    assert geff.shape[1] + start_pos[1] <= 64, "mapping exceed number of columns!"

    # map the geff to 64x64 array
    stor_vec = np.zeros((64, 64))
    stor_vec[start_pos[0]:start_pos[0] + geff.shape[0], start_pos[1]:start_pos[1] + geff.shape[1]] = geff
    # define devices which is within geff that are going to be written
    Msel = np.zeros((64, 64))
    # Msel = np.ones((64, 64))
    Msel[start_pos[0]:start_pos[0] + geff.shape[0], start_pos[1]:start_pos[1] + geff.shape[1]] = 1

    dpe.tune_conductance(array, stor_vec, Gtol_in=Gtol_in, Gtol_out=Gtol_out, Msel=Msel, method=method,
                         maxSteps=maxSteps, Tdly=Tdly, maxRetry=maxRetry, TwidthReset=Twidth, TwidthSet=Twidth,
                         vSetRamp=vSetRamp, vResetRamp=vResetRamp, vGateSetRamp=vGateSetRamp,
                         vGateResetRamp=vGateResetRamp)

    # read_conductance
    gmap = dpe.read(array, method=method)
    # transfer input to crossbar input, make sure xbarinput.shape[0] == 64
    if vec.shape[0] == 64:
        xbarinput = vec
    else:
        xbarinput = np.zeros((64, vec.shape[1]))
        xbarinput[start_pos[0]:start_pos[0] + vec.shape[0], :] = vec

    # linear correction
    if lincor:
        if xbarinput.min() < 0:
            lin_corr = np.zeros((geff.shape[1], 2))
            # rand_in = np.random.rand(vec.shape[0], 500)
            # testin = np.zeros((64, 500))
            # testin[start_pos[0]:start_pos[0] + vec.shape[0], :] = rand_in
            testin = xbarinput[:, :100]
            results_cali = (testin.T @ gmap).T
            results = dpe_pn(dpe, testin, array, [start_pos[1], start_pos[1] + geff.shape[1]], tdly=tdly).T
            for i in range(results.shape[0]):
                p1 = np.polyfit(results[i], results_cali[i + start_pos[1]], 1)
                lin_corr[i] = p1
        else:
            lin_corr = np.zeros((geff.shape[1], 2))
            # rand_in = np.random.rand(vec.shape[0], 500)
            # testin = np.zeros((64, 500))
            # testin[start_pos[0]:start_pos[0] + vec.shape[0], :] = rand_in
            testin = xbarinput[:, :100]
            results_cali = (testin.T @ gmap).T
            results = dpe.multiply(array, testin, c_sel=[start_pos[1], start_pos[1] + geff.shape[1]], Tdly=tdly).T
            for i in range(results.shape[0]):
                p1 = np.polyfit(results[i], results_cali[i + start_pos[1]], 1)
                lin_corr[i] = p1

    # perform matrix multiplication
    if xbarinput.min() < 0:
        output = dpe_pn(dpe, xbarinput, array, [start_pos[1], start_pos[1] + geff.shape[1]], tdly=tdly)
    else:
        output = dpe.multiply(array, xbarinput, c_sel=[start_pos[1], start_pos[1] + geff.shape[1]], r_start=0, mode=0,
                              Tdly=tdly)

    if lincor:
        output = dpe.lin_corr(output, lin_corr)

    return output, gmap[start_pos[0]:start_pos[0] + geff.shape[0], start_pos[1]:start_pos[1] + geff.shape[1]]


def xbar_mm(dpe, X, trgGmap, array, position, **kwargs):
    """

    :param dpe: DPE class for superT platform
    :param array: n-tuple with each int
                  which array wrt. tiled part of matrix
    :param X: <np.ndarray> of shape (r_x, c_x)
              input vectors, normalized or not normalized
    :param trgGmap: <np.ndarray> of shape (r_g, c_g)
                    target conductance map
    :param position: n-tuple with each (start_row, start_c)
                     position of tiled part in trgGmap to be mapped in 64x64 crossbar array
    :return:
            Y: <np.ndarray> of shape (c_x, c_g)
               output current, need to be transferred to actual output
            Gmap: <np.ndarray> of shape (r_g, c_g)
    """
    assert len(array) == len(position), "Array index should be specified wrt. each position"
    r_x, c_x = X.shape
    r_g, c_g = trgGmap.shape
    assert r_x == r_g, "First dimension doesn't match"
    # split into tiled arrays
    v_eff, g_eff, n_r, n_c = Xbar_tile_aggre(X, trgGmap)
    assert n_r * n_c == len(position), "Number of tiled arrays doesn't match"

    Y = []
    Gmap = np.zeros_like(trgGmap)
    for i in range(n_r):
        r_out = []
        for j in range(n_c):
            output, gmap = vec_mm(dpe, array[i * n_c + j], v_eff[f'v{i}'], g_eff[f'g{i}_{j}'], position[i * n_c + j],
                                  **kwargs)
            Gmap[i * 64:i * 64 + gmap.shape[0], j * 64:j * 64 + gmap.shape[1]] = gmap
            r_out.append(output)
        Y.append(np.concatenate(r_out, axis=-1))

    Y = np.array(Y).sum(axis=0)

    return Y, Gmap
