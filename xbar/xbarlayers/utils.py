"""Utils for crossbar layers including normalization, weight mapping"""

import numpy as np


def Norm(X, order=np.inf, axis=None):
    """
    Normalization for input arrays
    :param X: <np.ndarray> of shape (r, c)
              input or conductance map for crossbar arrays
    :param order: 1, 2, np.inf.
                  Norm type. 1 means l1 norm of vectors. 2 means l2 norm of vectors. np.inf means inf norm of vectors
                  Notice: Matrix norm is different from vector norm, we don't use matrix norm here.
    :param axis: int
                 along which dimension to be normalized
    :return:
            X_norm: <np.adarray> of shape (*, n_ex)
                    input vectors after normalization for voltage transfer
            norm: <np.ndarray> of shape (1, n_ex) or int
                  norm of type 'order' of a set of vectors
    """
    norm = np.linalg.norm(X, ord=order, axis=axis, keepdims=True)
    # For inf norm for matrix, return the maximum abs number in the matrix
    if order == np.inf and axis is None:
        norm = np.linalg.norm(X, ord=order, axis=0, keepdims=True)
        norm = np.linalg.norm(norm, ord=order, axis=axis, keepdims=True)
        assert norm == X.max(), "Inf norm of a matrix should return the maximum value in the matrix"
    X_norm = X / norm

    return X_norm, norm


#########################################################
#             Conductance mapping strategy              #
#########################################################

def dp_sparse(W, w_g_ratio, normalize=False, **kwargs):
    """
    Sparse differential pair method for weight mapping, half of the memrsitor is set to zeros.
    W is not normalized
    :param W: <np.ndarray> of shape (w_rows, w_cols)
              Weight parameters after transferring to 2D
    :param w_g_ratio: int
                      weight to conductance ratio
    :param normalize: bool
                      whether to normalize the weight mapping
    :return:
            trgGmap: <np.ndarray> of shape (w_rows, 2 * w_cols)
                     Conductance map of weight W
    """
    w_rows, w_cols = W.shape
    trgGmap = np.zeros((w_rows, 2 * w_cols))

    # Let the weight which is bigger than zero stay at the left column and rest right column
    W_pos, W_neg = W.copy(), W.copy()
    W_pos[W_pos < 0] = 0
    W_neg[W_neg > 0] = 0
    trgGmap[:, ::2] = W_pos
    trgGmap[:, 1::2] = -W_neg

    assert (trgGmap[:, ::2] - trgGmap[:, 1::2]).all() == W.all(), "Error when doing differential pair"

    if normalize:
        order = kwargs.get("order", np.inf)
        axis = kwargs.get("axis", None)
        trgGmap, norm = Norm(trgGmap, order, axis)

        return trgGmap * w_g_ratio, norm

    return trgGmap * w_g_ratio, 1.


def dp_dense(W, w_g_ratio, normalize=False, **kwargs):
    """
    Dense differential pair method for weight mapping.
    Left column: (w_offset + w) / 2; Right column: (w_offset - w) / 2; maximum value of the transferred weight.
    Usually use this mapping when doing online training, if so, don't need to be normalized
    :param W: <np.ndarray> of shape (w_rows, w_cols)
              Weight parameters after transferring to 2D
    :param w_g_ratio: int
                      weight to conductance ratio
    :param normalize: bool
                      Whether to normalize the weight after transferring, if True, then normalize the weight.
    :return:
            trgGmap: <np.ndarray> of shape (w_rows, 2 * w_cols)
    """
    # Set the w_offset to be the maximum abs value in matrix to ensure all transferred weights are bigger than zero
    # You can set different w_offset to fit in the crossbar dynamic range even <np.ndarray>
    w_offset = kwargs.get("w_offset", abs(W).max())
    w_rows, w_cols = W.shape
    trgGmap = np.zeros((w_rows, 2 * w_cols))

    # Set the left and right column
    trgGmap[:, ::2] = (w_offset + W) / 2
    trgGmap[:, 1::2] = (w_offset - W) / 2

    # Normalization for target conductance map, ususlly used in inference when the original weight is large
    if normalize:
        order = kwargs.get("order", np.inf)
        axis = kwargs.get("axis", None)
        trgGmap, norm = Norm(trgGmap, order, axis)
        return trgGmap * w_g_ratio, norm

    return trgGmap * w_g_ratio, 1.


def acm(W, w_g_ratio, normalize=False, **kwargs):
    """
    Adjacent column method for weight mapping, don't need to be normalized out of the function
    _______________________________________
    The first column of the conductance map
    need to be reconsidered
    _______________________________________
    :param W: <np.ndarray> of shape (w_rows, w_cols)
              Weight parameters after transferring to 2D
    :param w_g_ratio: int
                      weight to conductance ratio
    :param normalize: bool
                      whether to normalize the weight after transferring to weight mapping
    :return:
            trgGmap: <np.ndarray> of shape (w_rows, w_cols + 1)
                     Conductance map of weight W
            norm: <np.ndarray> of shape (1, w_rows)
    """
    w_rows, w_cols = W.shape
    trgGmap = np.zeros((w_rows, w_cols + 1))

    # shape of (w_rows, 1)
    _, firstcol = Norm(W, order=1, axis=1)
    # Set the first column to be the l1 norm of W, ensuring the conductances are all bigger than 0
    trgGmap[:, 0] = firstcol.reshape(-1)
    for i in range(w_cols):
        trgGmap[:, i + 1] = trgGmap[:, 0] - W[:, :i + 1].sum(axis=1)

    if normalize:
        # (w_rows, 1) or single value for normalization of conductance map
        order = kwargs.get("order", np.inf)
        axis = kwargs.get("axis", None)
        trgGmap, norm = Norm(trgGmap, order=order, axis=axis)

        return trgGmap * w_g_ratio, norm

    return trgGmap * w_g_ratio, 1.




#########################################################
#          Splitting matrix into tiled arrays           #
#########################################################

def Xbar_tile_aggre(v, g, r_size=64, c_size=64):
    """
    Split a matrix which extends array size to several tiled arrays to fit in the crossbar arrays
    param: v: vector to be multiplied, should be transposed to adapt to lib_dpe_utils [r, c_v]
    param: g: conductance map [r, c_g]
    output:
            dict_vec: n_r keys, values, split into n_r sets
            dict_gmap: n_r * n_c keys, values, split intp n_r * n_c sets
    """
    if np.ndim(v) == 1:
        v = np.expand_dims(v, 0)

    assert v.shape[0] == g.shape[0], "the dimension doesn't match!"
    n_r = np.ceil(g.shape[0] / r_size)  # number of tiled arrays in rows
    n_c = np.ceil(g.shape[1] / c_size)  # number of tiled arrays in columns

    dict_vec = {}
    dict_gmap = {}

    for i in range(n_r):
        if i == n_r - 1:
            dict_vec[f'v{i}'] = v[i * r_size:, :]
        else:
            dict_vec[f'v{i}'] = v[i * r_size:(i + 1) * r_size, :]

    for i in range(n_r):
        if i == n_r - 1:
            for j in range(n_c):
                if j == n_c - 1:
                    dict_gmap[f'g{i}_{j}'] = g[i * r_size:, j * c_size:]
                else:
                    dict_gmap[f'g{i}_{j}'] = g[i * r_size:, j * c_size:(j + 1) * c_size]
        else:
            for j in range(n_c):
                if j == n_c - 1:
                    dict_gmap[f'g{i}_{j}'] = g[i * r_size:(i + 1) * r_size, j * c_size:]
                else:
                    dict_gmap[f'g{i}_{j}'] = g[i * r_size:(i + 1) * r_size, j * c_size:(j + 1) * c_size]

    return dict_vec, dict_gmap, n_r, n_c