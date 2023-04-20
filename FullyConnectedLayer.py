import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * 0.01
        self.biases = np.zeros(out_features)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, d_out):
        d_x = np.dot(d_out, self.weights.T)
        d_weights = np.dot(self.input.T, d_out)
        d_biases = np.sum(d_out, axis=0)
        return d_x, d_weights, d_biases