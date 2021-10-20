import numpy as np


def Xbar_tile_aggre(v, g, r_size=64, c_size=64, **kwargs):
    """
    v: vector to be multiply, should be transposed to adapt to dpe [r, c_v]
    g: conductance map [r, c_g]
    output: [c_v, c_g]
    """
    if np.ndim(v) == 1:
        v = np.expand_dims(v, 0)

    assert v.shape[0] == g.shape[0], "the dimension doesn't match!"
    n_r = int((g.shape[0] - 1e-3) // r_size + 1)
    n_c = int((g.shape[1] - 1e-3) // c_size + 1)

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


def idx2onehot(idx, n):
    assert np.max(idx) < n, "number of labels extend!"
    onehot = np.zeros((len(idx), n))
    for i in range(len(idx)):
        onehot[i, idx[i]] = 1

    return onehot


def sample(latent_dim, num_samples, labels):
    z = np.random.randn(num_samples, latent_dim)
    y = idx2onehot(labels, 10)
    return np.concatenate((z, y), axis=-1)


def exp_sample(samples, latent_dim, num_samples, labels):
    assert len(samples) == num_samples * latent_dim, "dimension doesn't match"
    z = samples.reshape(num_samples, latent_dim)
    y = idx2onehot(labels, 10)
    return np.concatenate((z, y), axis=-1)


def relu(x):
    return x * (x > 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def batchnorm2d(batch, param):
    for i in range(batch.shape[1]):
        batch[:, i, :, :] = ((batch[:, i, :, :] - param[2][i]) / np.sqrt(param[3][i] + 1e-5)) * param[0][i] + param[1][
            i]

    return batch
