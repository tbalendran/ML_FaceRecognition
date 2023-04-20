class FlattenLayer:
    def __init__(self):
        pass

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)
