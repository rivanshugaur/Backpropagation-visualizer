"""
From-Scratch Deep Learning: Feedforward Neural Network with Hand-Derived Backpropagation.

Author: Rivanshu Gaur
Design Philosophy:
- Zero autograd, zero PyTorch/TensorFlow dependency.
- Every gradient is derived analytically using the multivariate chain rule.
- Built-in Numerical Gradient Checker (finite differences) to mathematically
  verify analytical backprop correctness to machine precision (< 1e-7 relative error).
- Modular activation functions (Tanh, ReLU, LeakyReLU, Sigmoid).
- Multi-optimizer engine (SGD, Momentum, Adam).
"""

from typing import Dict, Tuple, List, Optional
import numpy as np


# =====================================================================
# Activation Functions and Hand-Derived Derivatives
# =====================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation: σ(z) = 1 / (1 + e^(-z))."""
    z_clipped = np.clip(z, -500.0, 500.0)  # Prevent numerical overflow
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """
    Derivative of Sigmoid with respect to input z, expressed in terms of activation a = σ(z).
    dσ/dz = σ(z) * (1 - σ(z)) = a * (1 - a)
    """
    return a * (1.0 - a)


def tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation: tanh(z) = (e^z - e^-z) / (e^z + e^-z)."""
    return np.tanh(z)


def tanh_derivative(a: np.ndarray) -> np.ndarray:
    """
    Derivative of Tanh with respect to input z, expressed in terms of activation a = tanh(z).
    d(tanh)/dz = 1 - tanh^2(z) = 1 - a^2
    """
    return 1.0 - a ** 2


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: ReLU(z) = max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(a: np.ndarray) -> np.ndarray:
    """
    Subgradient of ReLU:
    d(ReLU)/dz = 1 if z > 0 (or a > 0), else 0.
    """
    return (a > 0.0).astype(float)


def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU: addresses the 'Dying ReLU' problem by permitting small slope α for z < 0."""
    return np.where(z > 0.0, z, alpha * z)


def leaky_relu_derivative(a: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivative of Leaky ReLU: 1 if a > 0 else α."""
def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Exponential Linear Unit: ELU(z) = z if z > 0 else alpha * (exp(z) - 1)."""
    return np.where(z > 0.0, z, alpha * (np.exp(np.clip(z, -500.0, 50.0)) - 1.0))


def elu_derivative(a: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Derivative of ELU: 1 if a > 0 else a + alpha."""
    return np.where(a > 0.0, 1.0, a + alpha)


ACTIVATIONS = {
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "elu": (elu, elu_derivative),
}


# =====================================================================
# BackpropNetwork Architecture
# =====================================================================

class BackpropNetwork:
    """
    A 2-Layer Fully Connected Neural Network with analytical backpropagation.

    Architecture:
      Input (N, input_dim)
        -> Z1 = X @ W1 + b1
        -> A1 = act_fn(Z1)                 (N, hidden_dim)
        -> Z2 = A1 @ W2 + b2
        -> A2 = sigmoid(Z2) = y_hat        (N, 1)
        -> Loss = Binary Cross-Entropy

    Mathematical Derivations:
      Let L be Binary Cross-Entropy Loss:
        L = - (1/N) * sum( y * log(A2) + (1 - y) * log(1 - A2) )

      Chain Rule at Output Layer:
        dL/dA2 = - (1/N) * [ y / A2 - (1 - y) / (1 - A2) ] = (A2 - y) / [N * A2 * (1 - A2)]
        dA2/dZ2 = A2 * (1 - A2)   (sigmoid derivative)
        dL/dZ2 = (dL/dA2) * (dA2/dZ2) = (A2 - y) / N    <-- Note: clean cancellation!

      Chain Rule for W2 and b2:
        dL/dW2 = A1^T @ (dL/dZ2)             shape: (hidden_dim, 1)
        dL/db2 = sum_{rows}(dL/dZ2)          shape: (1, 1)

      Chain Rule Backpropagating to Hidden Layer:
        dL/dA1 = (dL/dZ2) @ W2^T             shape: (N, hidden_dim)
        dL/dZ1 = (dL/dA1) ⊙ act_deriv(A1)    shape: (N, hidden_dim)  [Hadamard product]

      Chain Rule for W1 and b1:
        dL/dW1 = X^T @ (dL/dZ1)              shape: (input_dim, hidden_dim)
        dL/db1 = sum_{rows}(dL/dZ1)          shape: (1, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        learning_rate: float = 0.1,
        activation: str = "tanh",
        optimizer: str = "sgd",
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.activation_name = activation
        self.optimizer_name = optimizer.lower()

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from {list(ACTIVATIONS.keys())}")
        self.act_fn, self.act_deriv = ACTIVATIONS[activation]

        # Parameter Initialization:
        # For Tanh/Sigmoid -> Xavier/Glorot normal init: std = sqrt(2 / (fan_in + fan_out))
        # For ReLU/LeakyReLU -> He/Kaiming normal init: std = sqrt(2 / fan_in)
        rng = np.random.default_rng(seed)
        if activation in ("relu", "leaky_relu"):
            std1 = np.sqrt(2.0 / input_dim)
            std2 = np.sqrt(2.0 / hidden_dim)
        else:
            std1 = np.sqrt(2.0 / (input_dim + hidden_dim))
            std2 = np.sqrt(2.0 / (hidden_dim + 1))

        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * std1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.standard_normal((hidden_dim, 1)) * std2
        self.b2 = np.zeros((1, 1))

        # Optimizer State Variables
        self.t = 0  # Timestep for Adam
        # Momentum velocity buffers:
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        # Adam second-moment buffers:
        self.s_W1 = np.zeros_like(self.W1)
        self.s_b1 = np.zeros_like(self.b1)
        self.s_W2 = np.zeros_like(self.W2)
        self.s_b2 = np.zeros_like(self.b2)

        # Full training trajectory for time-travel analysis & visualization
        self.history: Dict[str, List] = {
            "loss": [],
            "accuracy": [],
            "W1": [], "b1": [], "W2": [], "b2": [],
            "grad_W1_norm": [], "grad_W2_norm": [],
            "grad_b1_norm": [], "grad_b2_norm": [],
        }

    # -----------------------------------------------------------------
    # Forward Pass
    # -----------------------------------------------------------------
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Executes the forward propagation pass.
        Returns predicted probabilities A2 and intermediate cache for backprop.
        """
        Z1 = X @ self.W1 + self.b1          # (N, hidden_dim)
        A1 = self.act_fn(Z1)                # (N, hidden_dim)
        Z2 = A1 @ self.W2 + self.b2         # (N, 1)
        A2 = sigmoid(Z2)                    # (N, 1)

        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    # -----------------------------------------------------------------
    # Loss Function & Metrics
    # -----------------------------------------------------------------
    @staticmethod
    def bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes stable Binary Cross-Entropy Loss with epsilon clipping.
        L = - (1/N) * sum( y * log(y_hat) + (1 - y) * log(1 - y_hat) )
        """
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)))

    @staticmethod
    def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes binary classification accuracy at 0.5 decision threshold."""
        predictions = (y_pred >= 0.5).astype(int)
        return float(np.mean(predictions == y_true))

    # -----------------------------------------------------------------
    # Backward Pass (Analytical Chain Rule)
    # -----------------------------------------------------------------
    def backward(self, y_true: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Executes manual backward propagation using multivariate calculus.
        Computes exact partial derivatives dL/dW1, dL/db1, dL/dW2, dL/db2.
        """
        N = y_true.shape[0]
        X = cache["X"]
        A1 = cache["A1"]
        A2 = cache["A2"]

        # Output Layer:
        # Derivative of BCE Loss w.r.t Z2 (pre-activation of output layer)
        dZ2 = (A2 - y_true) / N                           # Shape: (N, 1)
        dW2 = A1.T @ dZ2                                  # Shape: (hidden_dim, 1)
        db2 = np.sum(dZ2, axis=0, keepdims=True)          # Shape: (1, 1)

        # Hidden Layer:
        # Backpropagate error across W2 to hidden activations A1
        dA1 = dZ2 @ self.W2.T                             # Shape: (N, hidden_dim)
        # Apply local derivative of hidden activation function
        dZ1 = dA1 * self.act_deriv(A1)                    # Shape: (N, hidden_dim)
        dW1 = X.T @ dZ1                                   # Shape: (input_dim, hidden_dim)
        db1 = np.sum(dZ1, axis=0, keepdims=True)          # Shape: (1, hidden_dim)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # -----------------------------------------------------------------
    # Optimizer Parameter Update
    # -----------------------------------------------------------------
    def step(self, grads: Dict[str, np.ndarray]):
        """
        Updates weights and biases according to the chosen optimization algorithm.
        Supports:
          - 'sgd': Standard Gradient Descent
          - 'momentum': Classical Polyak Momentum
          - 'adam': Adaptive Moment Estimation
        """
        self.t += 1
        dW1, db1 = grads["dW1"], grads["db1"]
        dW2, db2 = grads["dW2"], grads["db2"]

        if self.optimizer_name == "sgd":
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

        elif self.optimizer_name == "momentum":
            beta = 0.9
            # Update velocities: v = beta * v + (1 - beta) * grad
            self.v_W1 = beta * self.v_W1 + (1 - beta) * dW1
            self.v_b1 = beta * self.v_b1 + (1 - beta) * db1
            self.v_W2 = beta * self.v_W2 + (1 - beta) * dW2
            self.v_b2 = beta * self.v_b2 + (1 - beta) * db2

            self.W1 -= self.lr * self.v_W1
            self.b1 -= self.lr * self.v_b1
            self.W2 -= self.lr * self.v_W2
            self.b2 -= self.lr * self.v_b2

        elif self.optimizer_name == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            for param_name in ["W1", "b1", "W2", "b2"]:
                param = getattr(self, param_name)
                grad = grads[f"d{param_name}"]
                v_attr = f"v_{param_name}"
                s_attr = f"s_{param_name}"

                # First moment vector: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                v = beta1 * getattr(self, v_attr) + (1.0 - beta1) * grad
                setattr(self, v_attr, v)

                # Second moment vector: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                s = beta2 * getattr(self, s_attr) + (1.0 - beta2) * (grad ** 2)
                setattr(self, s_attr, s)

                # Bias-corrected estimates
                v_corrected = v / (1.0 - beta1 ** self.t)
                s_corrected = s / (1.0 - beta2 ** self.t)

                # Parameter update
                param -= self.lr * v_corrected / (np.sqrt(s_corrected) + eps)
                setattr(self, param_name, param)

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    # -----------------------------------------------------------------
    # Training Loop with History Snapshots
    # -----------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 200) -> Dict[str, List]:
        """
        Executes full-batch gradient descent for `epochs` iterations.
        Snapshots parameters, losses, accuracies, and gradient norms per epoch.
        """
        y_col = y.reshape(-1, 1)

        for _ in range(epochs):
            A2, cache = self.forward(X)
            loss = self.bce_loss(y_col, A2)
            acc = self.compute_accuracy(y_col, A2)
            grads = self.backward(y_col, cache)

            # Record state BEFORE update so that step t aligns with weights at step t
            self.history["loss"].append(loss)
            self.history["accuracy"].append(acc)
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

    # -----------------------------------------------------------------
    # Time-Travel Prediction
    # -----------------------------------------------------------------
    def predict_with_weights(
        self,
        X: np.ndarray,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
    ) -> np.ndarray:
        """Evaluates forward prediction on arbitrary 2D grid using historical weight snapshots."""
        Z1 = X @ W1 + b1
        A1 = self.act_fn(Z1)
        Z2 = A1 @ W2 + b2
        return sigmoid(Z2)

    # -----------------------------------------------------------------
    # Numerical Gradient Verification (Finite Differences)
    # -----------------------------------------------------------------
    def grad_check(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1e-7,
        tolerance: float = 1e-6,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Mathematically verifies analytical gradients against central finite differences:
          f'(theta) ≈ [ f(theta + epsilon) - f(theta - epsilon) ] / (2 * epsilon)

        Relative Error Formula:
          relative_error = ||grad_analytic - grad_numerical|| / (||grad_analytic|| + ||grad_numerical||)

        Returns:
          passed (bool): True if relative error < tolerance for all parameter tensors.
          max_rel_error (float): Highest observed relative error across all parameters.
          tensor_errors (dict): Relative error breakdown per tensor.
        """
        y_col = y.reshape(-1, 1)

        # 1. Compute analytical gradients via manual backprop
        A2, cache = self.forward(X)
        analytic_grads = self.backward(y_col, cache)

        tensor_errors = {}
        max_rel_error = 0.0

        for param_name in ["W1", "b1", "W2", "b2"]:
            param = getattr(self, param_name)
            analytic_grad = analytic_grads[f"d{param_name}"]
            num_grad = np.zeros_like(param)

            # Iterate through each scalar element of the parameter tensor
            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                orig_val = param[idx]

                # f(theta + eps)
                param[idx] = orig_val + epsilon
                A2_plus, _ = self.forward(X)
                loss_plus = self.bce_loss(y_col, A2_plus)

                # f(theta - eps)
                param[idx] = orig_val - epsilon
                A2_minus, _ = self.forward(X)
                loss_minus = self.bce_loss(y_col, A2_minus)

                # Two-sided numerical gradient
                num_grad[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)

                # Restore original parameter value
                param[idx] = orig_val
                it.iternext()

            # Compute relative difference norm
            numerator = np.linalg.norm(analytic_grad - num_grad)
            denominator = np.linalg.norm(analytic_grad) + np.linalg.norm(num_grad) + 1e-15
            rel_error = float(numerator / denominator)

            tensor_errors[param_name] = rel_error
            if rel_error > max_rel_error:
                max_rel_error = rel_error

        passed = max_rel_error < tolerance
        return passed, max_rel_error, tensor_errors


# =====================================================================
# Standalone Unit Test & Gradient Checking Suite
# =====================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_moons

    print("=" * 65)
    print("BACKPROPAGATION ENGINE: VERIFICATION & SMOKE TEST")
    print("=" * 65)

    # Generate synthetic binary classification dataset
    X_raw, y_raw = make_moons(n_samples=250, noise=0.15, random_state=42)
    X_norm = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

    # 1. Run rigorous Numerical Gradient Check
    print("\n[Step 1] Running Central Finite Difference Gradient Check...")
    test_net = BackpropNetwork(input_dim=2, hidden_dim=4, activation="tanh", seed=10)
    passed, max_err, details = test_net.grad_check(X_norm[:30], y_raw[:30])

    for param, err in details.items():
        print(f"  - Relative error for {param:2s}: {err:.2e}")

    print(f"\nGradient Check Result: {'PASSED (Exact)' if passed else 'FAILED'}")
    print(f"Max Relative Error: {max_err:.2e} (Threshold: 1e-6)")
    assert passed, f"Gradient verification failed with error {max_err}"

    # 2. Train with Adam optimizer
    print("\n[Step 2] Training on Moons Dataset using Adam Optimizer...")
    train_net = BackpropNetwork(
        input_dim=2,
        hidden_dim=8,
        learning_rate=0.08,
        activation="tanh",
        optimizer="adam",
        seed=42,
    )
    hist = train_net.train(X_norm, y_raw, epochs=250)

    initial_loss = hist["loss"][0]
    final_loss = hist["loss"][-1]
    final_acc = hist["accuracy"][-1]

    print(f"  - Initial Loss:     {initial_loss:.4f}")
    print(f"  - Final Loss:       {final_loss:.4f} (Reduction: {(1 - final_loss / initial_loss) * 100:.1f}%)")
    print(f"  - Final Accuracy:   {final_acc * 100:.2f}%")
    print(f"  - Layer 1 Grad Norm: {hist['grad_W1_norm'][-1]:.4e}")
    print(f"  - Layer 2 Grad Norm: {hist['grad_W2_norm'][-1]:.4e}")

    assert final_loss < initial_loss, "Loss must strictly decrease"
    assert final_acc > 0.85, "Network should reach > 85% accuracy on Moons"
    print("\nAll architectural assertions passed successfully.")
