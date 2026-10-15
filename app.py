"""
Backpropagation Visualizer
---------------------------
An interactive Streamlit app with two intuitive views:

  1. Explore Trained Model — trains a from-scratch (NumPy-only) neural
     network and lets you scrub through training epoch-by-epoch to see
     the decision boundary, loss curve, gradient magnitudes, and weight
     matrices at any point in training.
  2. Watch It Learn (Animated) — a step-by-step animated walkthrough of
     one training example moving through forward pass, loss, backward
     pass, and weight update, in motion.

Run locally:
    streamlit run app.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from sklearn.datasets import make_moons, make_circles, make_blobs

from nn import BackpropNetwork

st.set_page_config(
    page_title="Backpropagation Visualizer",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------- Sidebar controls -----------------------------
st.sidebar.title("⚙️ Configuration")

dataset_name = st.sidebar.selectbox(
    "Dataset", ["Moons", "Circles", "Blobs (2-class)"]
)
noise = st.sidebar.slider("Noise", 0.0, 0.4, 0.15, 0.05)
hidden_dim = st.sidebar.slider("Hidden layer size", 2, 32, 8)
activation = st.sidebar.selectbox("Activation function", ["tanh", "relu"])
learning_rate = st.sidebar.slider("Learning rate", 0.01, 1.0, 0.2, 0.01)
epochs = st.sidebar.slider("Epochs to train", 20, 300, 150, 10)
train_clicked = st.sidebar.button("🚀 Train Network", use_container_width=True)

st.title("🧠 Backpropagation Visualizer")
st.caption(
    "A neural network trained with hand-derived backpropagation (pure NumPy, "
    "no autograd) — scrub through training to see exactly what each epoch "
    "changed and why."
)

# ----------------------------- Dataset generation ----------------------------
@st.cache_data
def get_dataset(name, noise_val):
    if name == "Moons":
        X, y = make_moons(n_samples=250, noise=noise_val, random_state=42)
    elif name == "Circles":
        X, y = make_circles(n_samples=250, noise=noise_val, factor=0.45, random_state=42)
    else:
        X, y = make_blobs(n_samples=250, centers=2, cluster_std=1.0 + noise_val * 3, random_state=42)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y


X, y = get_dataset(dataset_name, noise)

# ----------------------------- Train (on click) ----------------------------
if train_clicked or "history" not in st.session_state:
    net = BackpropNetwork(
        input_dim=2,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        activation=activation,
        optimizer="sgd",
    )
    history = net.train(X, y, epochs=epochs)

    st.session_state["net"] = net
    st.session_state["history"] = history
    st.session_state["X"] = X
    st.session_state["y"] = y

net = st.session_state["net"]
history = st.session_state["history"]
X = st.session_state["X"]
y = st.session_state["y"]
n_epochs = len(history["loss"])

# ----------------------------- Tabs ----------------------------
tab_explore, tab_animated = st.tabs(["📊 Explore Trained Model", "🎬 Watch It Learn (Animated)"])

with tab_explore:
    # ----------------------------- Epoch scrubber ----------------------------
    epoch = st.slider("Scrub to Epoch", 0, n_epochs - 1, n_epochs - 1)

    col1, col2 = st.columns([1.2, 1])

    # ----------------------------- Decision boundary plot ----------------------------
    with col1:
        st.subheader("Decision Boundary")
        W1 = history["W1"][epoch]
        b1 = history["b1"][epoch]
        W2 = history["W2"][epoch]
        b2 = history["b2"][epoch]

        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 160),
            np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 160),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = net.predict_with_weights(grid, W1, b1, W2, b2).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contourf(xx, yy, probs, levels=25, cmap="RdBu", alpha=0.75, vmin=0, vmax=1)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.2, linestyles="--")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="black", s=35)
        ax.set_title(f"Epoch {epoch} — Loss: {history['loss'][epoch]:.4f}")
        fig.colorbar(contour, ax=ax, label="P(class = 1)")
        st.pyplot(fig)

    # ----------------------------- Loss curve ----------------------------
    with col2:
        st.subheader("Loss Curve")
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        ax2.plot(history["loss"], color="#1f77b4", linewidth=2)
        ax2.axvline(epoch, color="#f43f5e", linestyle="--", linewidth=1.2)
        ax2.scatter([epoch], [history["loss"][epoch]], color="#f43f5e", zorder=5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Binary Cross-Entropy Loss")
        ax2.grid(True, alpha=0.25)
        st.pyplot(fig2)

        st.subheader("Gradient Magnitude by Layer")
        grad_labels = ["W1 (input→hidden)", "b1", "W2 (hidden→output)", "b2"]
        grad_values = [
            history["grad_W1_norm"][epoch],
            history["grad_b1_norm"][epoch],
            history["grad_W2_norm"][epoch],
            history["grad_b2_norm"][epoch],
        ]
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.barh(grad_labels, grad_values, color="#ff7f0e")
        ax3.set_xlabel("‖gradient‖ (L2 norm)")
        ax3.grid(True, alpha=0.25, axis="x")
        st.pyplot(fig3)

    # ----------------------------- Weight heatmaps ----------------------------
    st.subheader("Weight Matrices at This Epoch")
    w_col1, w_col2 = st.columns(2)

    with w_col1:
        fig4, ax4 = plt.subplots(figsize=(5, 2.5))
        im = ax4.imshow(history["W1"][epoch].T, cmap="coolwarm", aspect="auto")
        ax4.set_title("W1 (input → hidden)")
        ax4.set_xlabel("input feature")
        ax4.set_ylabel("hidden unit")
        fig4.colorbar(im, ax=ax4)
        st.pyplot(fig4)

    with w_col2:
        fig5, ax5 = plt.subplots(figsize=(5, 2.5))
        im2 = ax5.imshow(history["W2"][epoch], cmap="coolwarm", aspect="auto")
        ax5.set_title("W2 (hidden → output)")
        ax5.set_xlabel("output unit")
        ax5.set_ylabel("hidden unit")
        fig5.colorbar(im2, ax=ax5)
        st.pyplot(fig5)

with tab_animated:
    st.subheader("Watch a single training example learn, step by step")
    st.caption("Step through one training example: forward pass → loss → backward pass → weight update.")

    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            animated_html = f.read()
        components.html(animated_html, height=1300, scrolling=True)
    else:
        st.error("index.html not found.")
