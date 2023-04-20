import numpy as np

class ReLULayer:
    def __init__(self):
        pass

    def forward(self, x):
        self.input = x
        return np.maximum(x, 0)

    def backward(self, d_out):
        d_x = d_out * (self.input > 0)
        return d_x