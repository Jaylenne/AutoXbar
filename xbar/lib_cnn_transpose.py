import numpy as np


class CNNTranspose:
    """
    see https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8 for details
    """
    def __init__(self,
                 weight,
                 bias,
                 padding,
                 stride,
                 output_padding,
                 imginput,
                 Gmax):
        # input: [I, O, H, W]
        # bias: [O, ]
        # W: [O, H x W x I + 1] if bias = True
        if bias is None:
            self.W = np.swapaxes(np.flip(weight, (2, 3)), 0, 1).reshape(weight.shape[1], -1)
        else:
            self.W = np.append(np.swapaxes(np.flip(weight, (2, 3)), 0, 1).reshape(weight.shape[1], -1), np.expand_dims(bias, axis=1), axis=1)
        self.weight = weight
        self.aggregate_dim = weight.shape[0] * weight.shape[2] * weight.shape[3]  # I x H x W
        self.bias = bias
        self.img = imginput
        self.padding = padding
        self.output_padding = output_padding
        self.size = weight.shape[2]
        self.h = imginput.shape[2]
        self.stride = stride
        self.Gmax = Gmax

        # Convolutional layers after scaling
        self.convscal = self.norm(self.W.reshape(-1))
        self.conv = self.W * (1 / self.convscal)  # [N, H x W x C + 1] value is in [-1, 1]
        self.Gconv = self.weightclosedp(self.conv.T, Gmax)  # [H x W x C + 1, N]

        # Inputs after scal
        self.inputscal = self.norm(self.transferinput().reshape(-1))
        # [H x W x C +1, num_pixel**2 * N]
        self.cnn_input = (self.transferinput().reshape(-1, self.W.shape[-1])).T * (1 / self.inputscal)

    def weightclosedp(self, Wm, Gmax):
        Gconv = np.zeros((Wm.shape[0], Wm.shape[1] * 2))
        Gpos = Wm.copy()
        Gpos[Gpos < 0] = 0
        Gconv[:, ::2] = Gpos * Gmax
        Gneg = Wm.copy()
        Gneg[Gneg > 0] = 0
        Gconv[:, 1::2] = -Gneg * Gmax

        return Gconv

    def transferinput(self):
        """
        Interpolate input images into new images for transpose convolution
        see https://blog.csdn.net/qq_27261889/article/details/86304061
        """
        padding = self.size - self.padding - 1
        interpolate = self.stride - 1
        new_h = self.h + (self.stride - 1) * (self.h - 1) + 2 * padding + self.output_padding
        # [N, C, H', W']
        new_input = np.zeros((self.img.shape[0], self.img.shape[1], new_h, new_h))

        # [N, C, H, W]
        for i in range(self.h):
            for j in range(self.h):
                new_input[:, :, padding + i * (interpolate + 1), padding + j * (interpolate + 1)] = self.img[:, :, i, j]

        num_pixel = new_h - self.size + 1
        # [N, num_pixel**2, H x W x C +1]
        cnn_input = np.zeros((self.img.shape[0], num_pixel ** 2, self.W.shape[-1]))

        for i in range(num_pixel):
            for j in range(num_pixel):
                cnn_input[:, i * num_pixel + j, :self.aggregate_dim] = new_input[:, :, i:i + self.size,
                                                                       j:j + self.size].reshape(self.img.shape[0], -1)

        if self.bias is not None:
            cnn_input[:, :, -1] = 1

        return cnn_input

    def norm(self, vec):
        vec = vec.reshape(-1)
        vec_pos = vec.copy()
        vec_pos[vec_pos < 0] = 0

        vec_neg = vec.copy()
        vec_neg[vec_neg > 0] = 0

        scaling = max(vec_pos.max(), -vec_neg.min())

        return scaling

    def max_pooling(self, image_input, size=(2, 2)):
        x_dim = image_input.shape[1] // size[0]
        y_dim = image_input.shape[2] // size[1]

        ch_dim = image_input.shape[0]

        image_output = np.zeros((ch_dim, x_dim, y_dim))

        for ch in range(ch_dim):
            for x in range(x_dim):
                for y in range(y_dim):
                    img_patch = image_input[ch, x * size[0]:(x + 1) * size[0],
                                y * size[1]:(y + 1) * size[1]]
                    image_output[ch, x, y] = img_patch.max()

        return image_output


if __name__ == '__main__':
    testcnn = CNNTranspose(np.random.randn(32, 16, 3, 3), np.zeros((16,)), 0, 2, 0, np.random.randn(10, 32, 6, 6), 50e-6)
