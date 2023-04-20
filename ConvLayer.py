import numpy as np

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(out_channels)

    def forward(self, x):
        batch_size, in_channels, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    for i in range(0, h - self.kernel_size + 1, self.stride):
                        for j in range(0, w - self.kernel_size + 1, self.stride):
                            out[b, c_out, i // self.stride, j // self.stride] += np.sum(
                                x[b, c_in, i:i + self.kernel_size, j:j + self.kernel_size] * self.weights[c_out, c_in]
                            )
                out[b, c_out] += self.biases[c_out]
        
        self.input = x
        return out

    def backward(self, d_out):
        batch_size, in_channels, h, w = self.input.shape
        d_weights = np.zeros_like(self.weights)
        d_biases = np.zeros_like(self.biases)
        d_x = np.zeros_like(self.input)
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    for i in range(0, h - self.kernel_size + 1, self.stride):
                        for j in range(0, w - self.kernel_size + 1, self.stride):
                            d_weights[c_out, c_in] += d_out[b, c_out, i // self.stride, j // self.stride] * self.input[b, c_in, i:i + self.kernel_size, j:j + self.kernel_size]
                            d_x[b, c_in, i:i + self.kernel_size, j:j + self.kernel_size] += d_out[b, c_out, i // self.stride, j // self.stride] * self.weights[c_out, c_in]
                d_biases[c_out] += np.sum(d_out[b, c_out])
        
        return d_x, d_weights, d_biases

