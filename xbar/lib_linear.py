import numpy as np


class Linear:
    def __init__(self,
                 weight,
                 bias,
                 imginput,
                 Gmax):
        # weight: [out_dim, in_dim], bias: [out_dim, ]
        # self.W: [out_dim, in_dim + 1]
        if bias is None:
            self.W = weight
        else:
            self.W = np.append(weight, np.expand_dims(bias, axis=1), axis=1)

        self.Gmax = Gmax

        # define conductance map
        self.linearscal = self.norm(self.W.reshape(-1))
        self.linear = self.W * (1 / self.linearscal)
        self.Glinear = self.weightclosedp(self.linear.T, Gmax) # [in_dim + 1, 2 * out_dim]

        # define transferred input
        # input [N, in_dim]
        if bias is None:
            self.img = imginput
        else:
            self.img = np.append(imginput, np.ones((imginput.shape[0], 1)), axis=1) # [N, in_dim + 1]
        self.inputscal = self.norm(self.img.reshape(-1))
        self.linear_input = (self.img.T * (1 / self.inputscal))



    def weightclosedp(self, Wm, Gmax):
        Gconv = np.zeros((Wm.shape[0], Wm.shape[1] * 2))
        Gpos = Wm.copy()
        Gpos[Gpos < 0] = 0
        Gconv[:, ::2] = Gpos * Gmax
        Gneg = Wm.copy()
        Gneg[Gneg > 0] = 0
        Gconv[:, 1::2] = -Gneg * Gmax

        return Gconv

    def norm(self, vec):
        vec = vec.reshape(-1)
        vec_pos = vec.copy()
        vec_pos[vec_pos < 0] = 0

        vec_neg = vec.copy()
        vec_neg[vec_neg > 0] = 0

        scaling = max(vec_pos.max(), -vec_neg.min())

        return scaling

if __name__ == "__main__":
    testlinear = Linear(np.random.randn(10, 5), np.random.randn(10), np.random.randn(2, 5), 50e-6)
    print(testlinear.linear.max())