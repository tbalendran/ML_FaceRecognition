import numpy as np

class MaxPoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        out = np.zeros((batch_size, channels, out_h, out_w))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, h - self.pool_size + 1, self.stride):
                    for j in range(0, w - self.pool_size + 1, self.stride):
                        out[b, c, i // self.stride, j // self.stride] = np.max(x[b, c, i:i + self.pool_size, j:j + self.pool_size])

        self.input = x
        return out

    def backward(self, d_out):
        batch_size, channels, h, w = self.input.shape
        d_x = np.zeros_like(self.input)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, h - self.pool_size + 1, self.stride):
                    for j in range(0, w - self.pool_size + 1, self.stride):
                        window = self.input[b, c, i:i + self.pool_size, j:j + self.pool_size]
                        max_val = np.max(window)
                        mask = (window == max_val)
                        d_x[b, c, i:i + self.pool_size, j:j + self.pool_size] += mask * d_out[b, c, i // self.stride, j // self.stride]

        return d_x

