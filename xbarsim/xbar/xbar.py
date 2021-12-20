import numpy as np

from .simArrayPy import simArrayPy
from ...xbar.utils import Xbar_tile_aggre


#
# Basic class of crossbar simulation by

class xbar:
    def __init__(self,
                 r_size=64,
                 c_size=64,
                 relax_noise_big=15.,
                 relax_noise_low=5.,
                 g1=20.,
                 g2=80.,
                 slope=0.782,
                 intercept=-2.168,
                 scale=1.,
                 rw=1e-6,
                 t=20e-9,
                 wm_type="dp_sparse",
                 w_g_ratio=150.,
                 Vread=0.2,
                 piecewise=True,
                 geff=False,
                 fluc_noise=False,
                 ):
        self.r_size = r_size
        self.c_size = c_size
        self.a = relax_noise_big
        self.b = relax_noise_low
        self.g1 = g1
        self.g2 = g2
        self.slope = slope
        self.intercept = intercept
        self.scale = scale
        self.rw = rw
        self.piecewise = piecewise
        self.geff = geff
        self.fluc_noise = fluc_noise
        self.t = t
        self.wm_type = wm_type
        self.w_g_ratio = w_g_ratio
        self.Vread = Vread

        # Energy Storage
        self.power = None
        self.Gmap = None

    def _cal_gmap(self, Ginit):
        Gmap = add_program_noise(Ginit, a=self.a, b=self.b, g1=self.g1, g2=self.g2, piecewise=self.piecewise)
        if self.geff:
            _, dict_gmap, n_r, n_c = Xbar_tile_aggre(np.random.randn(Ginit.shape[0], 1), Gmap,
                                                     r_size=self.r_size, c_size=self.c_size)
            for key, value in dict_gmap.items():
                dict_gmap[key] = cal_geff(value, rw=self.rw)
            Gmap = np.concatenate(
                [
                    np.concatenate(
                        [dict_gmap[f'g{i}_{j}'] for i in range(n_r)]
                    ) for j in range(n_c)
                ]
            )
        return Gmap

    def mm(self, vvec, G):
        """
        Matrix multiplication simulation for crossbar array

        Params:
        _________________
        vvec: input voltage vector, should be normalized and multiplied by the voltage [N, r]
        G: initial conductance map, should be normalized and within the dynamic range, unit: uS [r, c]

        Return:
        _________________
        I: output current
        """
        assert vvec.shape[1] == G.shape[0]
        Current = np.zeros((vvec.shape[0], G.shape[1]))
        self.power = 0

        Gmap = self._cal_gmap(G)
        if self.fluc_noise:
            for i, v in enumerate(vvec):
                # v: [r, ]
                Gmap_fluc = add_fluc_noise(Gmap, self.slope, self.intercept, self.scale)  # [r, c]
                Current[i] = (v[:, None] * Gmap_fluc).sum(axis=0)
                self.power += ((v[:, None] ** 2) * Gmap_fluc).sum()
            self.Gmap = Gmap_fluc
        else:
            Current[:] = np.dot(vvec, Gmap)
            self.power += np.dot(vvec ** 2, Gmap).sum()
            self.Gmap = Gmap

        return Current


def add_program_noise(Ginit, a=15., b=5., g1=20., g2=80., piecewise=True):
    """
    Add program error to the initial conductance of crossbar arrays

    Params:
    __________
    Ginit: Initial conductance to be programmed
    a: S.d. for g1 < conductance < g2
    b: S.d. for rest conductance range
    g1: Conductance bound 1
    g2: Conductance bound 2
    piecewise: If use piecewise function to characterize, otherwise use a for all conductance range

    Return:
    __________
    Gmap: Conductance map with the same size
    """
    assert Ginit.max() < 200, "Conductance range exceed"

    Gmap = np.zeros_like(Ginit)
    Gmap = np.abs(Gmap + np.random.randn(*Ginit.shape) * a)
    if piecewise:
        idx = np.where((Ginit < g1) | (Ginit > g2))
        G = Ginit[idx]
        Gmap[idx] = np.abs(G + np.random.randn(*G.shape) * b)

    return Gmap


def cal_geff(G, rw=1e-6):
    """
    Calculate effective conductance including wire resistance
    """
    simArray = simArrayPy(G, rw=rw)
    return simArray.geff


def add_fluc_noise(Ginit, slope=0.782, intercept=-2.168, scale=1.):
    """
    Add fluctuation noise for every input
    """
    EPS = 1e-15
    Gmap = np.zeros_like(Ginit)
    idx1 = np.where(Ginit < 50.)
    idx2 = np.where(Ginit >= 50.)
    G1 = Ginit[idx1]
    G2 = Ginit[idx2]

    loc1 = np.log(G1 + EPS) * slope + intercept
    loc2 = np.log(50.) * slope + intercept

    drift1 = np.clip(np.exp(loc1 + scale * np.random.randn(*G1.shape)), a_max=20., a_min=None)
    drift2 = np.clip(np.exp(loc2 + scale * np.random.randn(*G2.shape)), a_max=20., a_min=None)

    Gmap[idx1] = np.abs(G1 + np.random.randn(*G1.shape) * drift1)
    Gmap[idx2] = np.abs(G2 + np.random.randn(*G2.shape) * drift2)

    return Gmap


if __name__ == "__main__":
    xbar1 = xbar(geff=False, fluc_noise=True)
    x = np.random.randn(1000, 64)
    G = np.random.rand(64, 170) * 100
    xbar1.mm(x, G)
    print(xbar1.Gmap.max())