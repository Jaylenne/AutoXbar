"""Neural network operations"""
import numpy as np


#########################################################
#                 Padding operations                    #
#########################################################
def pad2D(X, pad, kernel_shape=None, stride=None):
    """
    Zero-pad a 4D input volume X along last two dimension
    :param X: <np.ndarray> of shape (n_ex, in_ch, in_rows, in_cols)
    :param pad: tuple, int, or 'same'
                The padding amount. If 'same', add padding so that the 2D convolution with 'kernel_shape' and 'stride'
                has the same dimension as the input, only support stride=1. If int, meaning add the same number of
                zeros to the top/bottom/left/right. If tuple, specifies the number of padding rows and colums to add
                on both sides of the rows/columns in `X`. If 4-tuple, specifies the number of rows/columns to add to
                the top, bottom, left, and right of the input volume.
    :param kernel_shape: 2-tuple
                         Dimension of 2D convolutional kernel
    :param stride: int
                   The stride for the convolutional kernel
    :return:
            X_pad: <np.ndarray> of shape (n_ex, in_ch, padded_in_rows, padded_out_rows)
                   The output volume after padding
            p: 4-tuple
               The number of 0-padded rows added to (top, bottom, left, right) of 'X'
    """
    p = pad
    if isinstance(p, int):
        p = (p, p, p, p)

    if isinstance(p, tuple):
        if len(p) == 2:
            p = (p[0], p[0], p[1], p[1])

    if p == 'same':
        k_r, k_c = kernel_shape
        pad_h = (np.ceil((k_r - 1) / 2), np.floor((k_c - 1) / 2))
        pad_w = (np.ceil((k_c - 1) / 2), np.floor((k_c - 1) / 2))
        p = (pad_h[0], pad_h[1], pad_w[0], pad_w[1])

    X_pad = np.pad(
        X,
        pad_width=((0, 0), (0, 0), (p[0], p[1]), (p[2], p[3])),
        mode='constant',
        constant_values=0,
    )

    return X_pad, p


def dilate(X, d):
    """
    Dilate the 4D volume 'X' by 'd'. Used for transpose 2D convolution
    :param X: <np.ndarray> of shape (n_ex, in_ch, in_rows, in_cols)
    :param d: int
              The number of 0-rows to insert between adjacent row and column in 'X'
    :return:
            Xd: <np.array> of shape (n_ex, in_ch, dilated_in_rows, dilated_in_cols)
                The dilated array
    """
    _, _, in_rows, in_cols = X.shape
    r_idx = np.repeat(np.arange(1, in_rows), d)
    c_idx = np.repeat(np.arange(1, in_cols), d)
    Xd = np.insert(X, r_idx, 0, axis=2)
    Xd = np.insert(Xd, c_idx, 0, axis=3)

    return Xd


def _im2col_indices(X_shape, fr, fc, p, s):
    """
    Helper function to calculate the indices of rearranged volume for 2D convolution wrt. original indices
    :param X_shape: 4-tuple of shape (n_ex, in_ch, in_rows, in_cols)
                    Shape of input volume
    :param fr: int
               Number of rows of 2D convolution kernel
    :param fc: int
               Number of columns of convolution kernel
    :param p: 4-tuple
              padding number
    :param s: int
              stride
    :return:
            indices of last three dimension wrt. original indices of 'X'
    """
    pr1, pr2, pc1, pc2 = p
    n_ex, in_ch, in_rows, in_cols = X_shape

    # compute the output number of rows and columns
    out_rows = (in_rows + pr1 + pr2 - fr) // s + 1
    out_cols = (in_cols + pc1 + pc2 - fc) // s + 1

    if any([out_rows <= 0, out_cols <= 0]):
        raise ValueError(
            "Dimension mismatch during convolution"
            "out_rows = {}, out_cols = {}".format(out_rows, out_cols)
        )

    i0 = np.tile(np.repeat(np.arange(fr), fc), in_ch)
    i1 = np.repeat(np.arange(out_rows), out_cols) * s
    j0 = np.tile(np.arange(fc), fr * in_ch)
    j1 = np.tile(np.arange(out_cols), out_rows) * s

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)  # (fr * fc * in_ch, out_rows * out_cols)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)  # (fr * fc * in_ch, out_rows * out_cols)
    k = np.repeat(np.arange(in_ch), fr * fc).reshape(-1, 1)  # (fr * fc * in_ch, 1)

    return k, i, j


def im2col(X, W_shape, pad, stride):
    """
    Transfer the 4D input X to the 2D vectors with rearranging the elements and concatenation
    for convolution operations.
    Similar transformation for crossbar array input
    :param X: <np.ndarray> of shape (n_ex, in_ch, in_rows, in_cols)
              input 4D volumes
    :param W_shape: 4-tuple of shape (out_ch, in_ch, k_rows, k_cols)
                    Shape of 2D convolution kernel
    :param pad: int, tuple, 'same'
                padding of 0-rows to the last two dimension of input
    :param stride: int
                   stride of 2D convolution operation
    :return:
            X_col: <np.ndarray> of shape (k_rows * k_cols * in_ch, n_ex * out_rows * out_cols)
    """
    n_ex, in_ch, in_rows, in_cols = X.shape
    out_ch, in_ch, fr, fc = W_shape
    p, s = pad, stride

    X_pad, p = pad2D(X, p, (fr, fc), s)

    k, i, j = _im2col_indices((n_ex, in_ch, in_rows, in_cols), fr, fc, p, s)

    # advanced indexing with size (fr * fc * in_ch, n_ex * out_rows * out_cols)
    X_col = X_pad[:, k, i, j]  # (n_ex, fr * fc * in_ch, out_rows * out_cols)
    X_col = X_col.transpose(1, 2, 0).reshape(fr * fc * in_ch, -1)

    return X_col


def col2im(X_col, X_shape, W_shape, pad, stride):
    """
    Take column representation of 2D X_col and rearrange it into the volume of 4D image volume. Used for
    backward process
    :param X_col: <np.ndarray> of shape (fr * fc * in_ch, out_rows * out_cols * n_ex)
                  Column representation of X
    :param X_shape: 4-tuple of shape (n_ex, in_ch, in_rows, in_cols)
                    Transferred 4D volume
    :param W_shape: 4-tuple of shape (out_ch, in_ch, k_rows, k_cols)
                    Shape of 4D convolutional kernel
    :param pad: 4-tuple of (top, bottom, left, right)
    :param stride: int
    :return:
             X_pad: <np.ndarray> of shape (n_ex, in_ch, in_rows, in_cols)
                4D volume input
    """
    if not (isinstance(pad, tuple) and len(pad) == 4):
        raise TypeError("Pad must be a 4-tuple, but got: {}".format(pad))

    assert X_shape[1] == W_shape[1], "Dimension of in_channel should be the same"

    s = stride
    pr1, pr2, pc1, pc2 = pad
    out_ch, in_ch, fr, fc = W_shape
    n_ex, in_ch, in_rows, in_cols = X_shape

    # The padded input for original 2D convolution process
    X_pad = np.zeros((n_ex, in_ch, in_rows + pr1 + pr2, in_cols + pc1 + pc2))
    k, i, j = _im2col_indices(X_shape, fr, fc, pad, stride)

    X_col_reshaped = X_col.reshape(fr * fc * in_ch, -1, n_ex)  # (fr * fc * in_ch, out_rows * out_cols, n_ex)
    X_col_reshaped = X_col_reshaped.tranpose(2, 0, 1)  # (n_ex, fr * fc * in_ch, out_rows * out_cols)

    # See https://stackoverflow.com/questions/45473896/np-add-at-indexing-with-array for details
    # reversed operation of im2col, fill the 0-volume X_pad with the element of X_col that mapped from X
    np.add.at(X_pad, (slice(None), k, i, j), X_col_reshaped)

    pr2 = None if pr2 == 0 else -1
    pc2 = None if pc1 == 0 else -1

    return X_pad[:, :, pr1:pr2, pc1:pc2]


#########################################################
#                 Convolution operations                #
#########################################################
def conv2D(X, W, pad, stride):
    """
    2D convolution layer with im2col
    :param X: <np.ndarray> of shape (n_ex, in_ch, in_rows, in_cols)
              Input 4D volume 'X'
    :param W: <np.ndarray> of shape (out_ch, in_ch, k_rows, k_cols)
            2D convolutional kernel
    :param pad: int, tuple, 'same'
                padding 0-rows to top/bottom/left/right of the last two dimensions of input 'X'
    :param stride: int
    :return:
            X_out: <np.ndarray> of shape (n_ex, out_ch, out_rows, out_cols)
    """
    s = stride
    out_ch, in_ch, fr, fc = W.shape
    n_ex, in_ch, in_rows, in_cols = X.shape

    _, p = pad2D(X, pad, (fr, fc), s)
    pr1, pr2, pc1, pc2 = p

    out_rows = (in_rows + pr1 + pr2 - fr) // s + 1
    out_cols = (in_cols + pc1 + pc2 - fc) // s + 1

    # X_col shape = (k_rows * k_cols * in_ch, n_ex * out_rows * out_cols)
    X_col = im2col(X, W.shape, p, s)
    W_col = W.reshape(out_ch, -1)  # (out_ch, k_rows * k_cols * in_ch)

    X_out = W_col.dot(X_col).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 0, 1, 2)

    return X_out
