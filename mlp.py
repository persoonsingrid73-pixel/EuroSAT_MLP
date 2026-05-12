import numpy as np
from models.layers import Linear, ReLU, Sigmoid, Tanh, CrossEntropyLoss, Dropout

class ThreeLayerMLP:
    def __init__(self, input_dim=12288, hidden_dim=512, num_classes=10, l2_reg=0.01, activation='tanh', dropout_p=0.5):
        self.l2_reg = l2_reg
        self.activation_name = activation.lower()   # 👈 必须绑定，用于保存配置
        self.dropout_p = dropout_p                  # 👈 必须绑定，用于保存配置

        # 自动判断初始化策略
        init_type = 'xavier' if self.activation_name in ['tanh', 'sigmoid'] else 'he'

        self.fc1 = Linear(input_dim, hidden_dim, init_method=init_type)
        self.fc2 = Linear(hidden_dim, hidden_dim // 2, init_method=init_type)
        self.fc3 = Linear(hidden_dim // 2, num_classes, init_method=init_type)

        if self.activation_name == 'relu':
            self.act1, self.act2 = ReLU(), ReLU()
        elif self.activation_name == 'sigmoid':
            self.act1, self.act2 = Sigmoid(), Sigmoid()
        elif self.activation_name == 'tanh':
            self.act1, self.act2 = Tanh(), Tanh()
        else:
            raise ValueError("Unsupported activation function.")

        self.dropout1 = Dropout(p=dropout_p)
        self.dropout2 = Dropout(p=dropout_p)

        self.loss_fn = CrossEntropyLoss()
        self.weight_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, X, training=True):
        out = self.fc1.forward(X)
        out = self.act1.forward(out)
        out = self.dropout1.forward(out, training=training)

        out = self.fc2.forward(out)
        out = self.act2.forward(out)
        out = self.dropout2.forward(out, training=training)

        out = self.fc3.forward(out)
        return out

    def compute_loss(self, logits, y):
        data_loss = self.loss_fn.forward(logits, y)
        reg_loss = 0.0
        if self.l2_reg > 0:
            for layer in self.weight_layers:
                reg_loss += 0.5 * self.l2_reg * np.sum(layer.W ** 2)
        return data_loss + reg_loss

    def backward(self):
        d_out = self.loss_fn.backward()
        d_out = self.fc3.backward(d_out)

        d_out = self.dropout2.backward(d_out)
        d_out = self.act2.backward(d_out)
        d_out = self.fc2.backward(d_out)

        d_out = self.dropout1.backward(d_out)
        d_out = self.act1.backward(d_out)
        d_out = self.fc1.backward(d_out)

        if self.l2_reg > 0:
            for layer in self.weight_layers:
                layer.dW += self.l2_reg * layer.W

    def step(self, learning_rate, momentum=0.9):
        for layer in self.weight_layers:
            layer.v_W = momentum * layer.v_W + (1.0 - momentum) * layer.dW
            layer.v_b = momentum * layer.v_b + (1.0 - momentum) * layer.db
            layer.W -= learning_rate * layer.v_W
            layer.b -= learning_rate * layer.v_b

    def predict(self, X):
        logits = self.forward(X, training=False)
        return np.argmax(logits, axis=1)