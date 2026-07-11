"""
From-scratch feedforward neural network with manual backpropagation.

No autograd, no PyTorch/TensorFlow — every gradient here is derived and
computed by hand using the chain rule. This is the core engine that powers
the Streamlit visualizer in app.py.

Architecture: Input -> Hidden (tanh or ReLU) -> Output (sigmoid), trained
with binary cross-entropy loss via full-batch gradient descent.
"""

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    # a is already sigmoid(z)
    return a * (1 - a)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(a):
    # a is already tanh(z)
    return 1 - a ** 2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(a):
    # a is already relu(z); derivative is 1 where a > 0, else 0
    return (a > 0).astype(float)


ACTIVATIONS = {
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
}


class BackpropNetwork:
    """
    A 2-layer (1 hidden layer) fully-connected neural network trained with
    hand-derived backpropagation. Records full training history so every
    epoch can be replayed and visualized later.
    """

    def __init__(self, input_dim, hidden_dim, learning_rate=0.1,
                 activation="tanh", seed=42):
        rng = np.random.default_rng(seed)
        # He/Xavier-style small random init
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(1 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.standard_normal((hidden_dim, 1)) * np.sqrt(1 / hidden_dim)
        self.b2 = np.zeros((1, 1))

        self.lr = learning_rate
        self.act_fn, self.act_deriv = ACTIVATIONS[activation]

        # Training history, one entry appended per epoch
        self.history = {
            "loss": [],
            "W1": [], "b1": [], "W2": [], "b2": [],
            "grad_W1_norm": [], "grad_W2_norm": [],
            "grad_b1_norm": [], "grad_b2_norm": [],
        }

    def forward(self, X):
        """Forward pass. Returns all intermediate activations (needed for backprop)."""
        Z1 = X @ self.W1 + self.b1          # (N, hidden_dim)
        A1 = self.act_fn(Z1)                # (N, hidden_dim)
        Z2 = A1 @ self.W2 + self.b2         # (N, 1)
        A2 = sigmoid(Z2)                    # (N, 1)  <- predicted probability
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    @staticmethod
    def bce_loss(y_true, y_pred):
        eps = 1e-8  # numerical stability
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, cache):
        """
        Manual backpropagation via the chain rule.

        Forward graph:  X -> Z1 -> A1 -> Z2 -> A2 -> Loss
        We compute dLoss/dA2, then walk backwards through each node,
        multiplying local derivatives (the chain rule) to get the gradient
        of the loss with respect to every weight and bias.
        """
        N = y_true.shape[0]
        X, A1, A2 = cache["X"], cache["A1"], cache["A2"]

        # --- Output layer ---
        # For BCE loss + sigmoid output, dL/dZ2 simplifies beautifully to (A2 - y)
        dZ2 = (A2 - y_true) / N                      # (N, 1)
        dW2 = A1.T @ dZ2                              # (hidden_dim, 1)
        db2 = np.sum(dZ2, axis=0, keepdims=True)      # (1, 1)

        # --- Hidden layer (chain rule back through W2, then activation) ---
        dA1 = dZ2 @ self.W2.T                          # (N, hidden_dim)
        dZ1 = dA1 * self.act_deriv(A1)                  # element-wise, chain rule
        dW1 = X.T @ dZ1                                 # (input_dim, hidden_dim)
        db1 = np.sum(dZ1, axis=0, keepdims=True)        # (1, hidden_dim)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def step(self, grads):
        """Gradient descent update: theta = theta - lr * grad."""
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]

    def train(self, X, y, epochs=200):
        """Full training loop. Snapshots weights + gradient norms every epoch."""
        y = y.reshape(-1, 1)
        for _ in range(epochs):
            A2, cache = self.forward(X)
            loss = self.bce_loss(y, A2)
            grads = self.backward(y, cache)

            # Record history BEFORE updating, so epoch i shows the weights
            # that produced epoch i's loss, and the gradients that will move
            # it to epoch i+1.
            self.history["loss"].append(float(loss))
            self.history["W1"].append(self.W1.copy())
            self.history["b1"].append(self.b1.copy())
            self.history["W2"].append(self.W2.copy())
            self.history["b2"].append(self.b2.copy())
            self.history["grad_W1_norm"].append(float(np.linalg.norm(grads["dW1"])))
            self.history["grad_W2_norm"].append(float(np.linalg.norm(grads["dW2"])))
            self.history["grad_b1_norm"].append(float(np.linalg.norm(grads["db1"])))
            self.history["grad_b2_norm"].append(float(np.linalg.norm(grads["db2"])))

            self.step(grads)

        return self.history

    def predict_with_weights(self, X, W1, b1, W2, b2):
        """Run a forward pass using a specific historical snapshot of weights."""
        Z1 = X @ W1 + b1
        A1 = self.act_fn(Z1)
        Z2 = A1 @ W2 + b2
        return sigmoid(Z2)


if __name__ == "__main__":
    # Quick smoke test: does the network actually learn a simple XOR-like
    # pattern? Loss should decrease substantially over training.
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # normalize

    net = BackpropNetwork(input_dim=2, hidden_dim=8, learning_rate=0.5, activation="tanh")
    history = net.train(X, y, epochs=300)

    print(f"Initial loss: {history['loss'][0]:.4f}")
    print(f"Final loss:   {history['loss'][-1]:.4f}")

    preds = (net.forward(X)[0] > 0.5).astype(int).ravel()
    acc = (preds == y).mean()
    print(f"Training accuracy: {acc * 100:.1f}%")

    assert history["loss"][-1] < history["loss"][0], "Loss should decrease"
    assert acc > 0.85, "Should learn moons dataset reasonably well"
    print("Smoke test passed.")
