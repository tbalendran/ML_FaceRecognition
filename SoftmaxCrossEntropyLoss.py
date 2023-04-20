import numpy as np

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        pass

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, logits, labels):
        batch_size = logits.shape[0]
        self.probs = self.softmax(logits)
        self.labels = labels

        log_probs = -np.log(self.probs[np.arange(batch_size), labels])
        loss = np.sum(log_probs) / batch_size
        return loss

    def backward(self):
        batch_size = self.probs.shape[0]
        d_logits = self.probs.copy()
        d_logits[np.arange(batch_size), self.labels] -= 1
        d_logits /= batch_size
        return d_logits