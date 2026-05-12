import numpy as np

class Linear:
    def __init__(self, in_features, out_features, init_method='he'):
        # 根据激活函数动态选择最优初始化策略
        if init_method == 'xavier':
            # Xavier/Glorot: 适合 Tanh 和 Sigmoid
            self.W = np.random.randn(in_features, out_features) * np.sqrt(1.0 / in_features)
        else:
            # He: 适合 ReLU
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
            
        self.b = np.zeros((1, out_features))
        self.X = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ):
        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        return np.dot(dZ, self.W.T)

class ReLU:
    def __init__(self): self.X = None
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)
    def backward(self, dOut):
        dX = dOut.copy()
        dX[self.X <= 0] = 0
        return dX

class Sigmoid:
    def __init__(self): self.out = None
    def forward(self, X):
        X_safe = np.clip(X, -500, 500)
        self.out = 1.0 / (1.0 + np.exp(-X_safe))
        return self.out
    def backward(self, dOut):
        return dOut * self.out * (1.0 - self.out)

class Tanh:
    def __init__(self): self.out = None
    def forward(self, X):
        X_safe = np.clip(X, -250, 250)
        self.out = np.tanh(X_safe)
        return self.out
    def backward(self, dOut):
        return dOut * (1.0 - self.out ** 2)

class CrossEntropyLoss:
    def __init__(self):
        self.Y_pred = None
        self.Y_true = None
    def forward(self, logits, Y_true):
        self.Y_true = Y_true
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        self.Y_pred = exps / np.sum(exps, axis=1, keepdims=True)
        N = logits.shape[0]
        return np.sum(-np.log(self.Y_pred[range(N), Y_true] + 1e-9)) / N
    def backward(self):
        N = self.Y_true.shape[0]
        d_logits = self.Y_pred.copy()
        d_logits[range(N), self.Y_true] -= 1
        d_logits /= N
        return d_logits

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self.mask
        else:
            return x
    def backward(self, dout):
        return dout * self.mask