"""
Backpropagation Visualizer (Streamlit Edition)
----------------------------------------------
An interactive Streamlit application powered by a from-scratch (pure NumPy, zero autograd)
neural network engine with analytical backpropagation.

Key Features:
  1. Explore Trained Model — scrub through training epoch-by-epoch to see decision
     boundaries, loss curves, accuracy, gradient L2 norms, and weight matrices.
  2. Numerical Finite Difference Gradient Check — mathematically verify analytical
     gradients against machine-precision central differences.
  3. Interactive Web Suite (Animated) — embeds the high-performance HTML5/JS
     visualizer with interactive computation graph and step-by-step backprop debugger.

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
    page_title="Backpropagation Visualizer — Analytical Neural Network",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Sidebar controls -----------------------------
st.sidebar.title("⚙️ Hyperparameters")

dataset_name = st.sidebar.selectbox(
    "Dataset Topology", ["Moons", "Circles", "Blobs (2-class)"]
)
noise = st.sidebar.slider("Noise level", 0.0, 0.4, 0.15, 0.05)
hidden_dim = st.sidebar.slider("Hidden layer size", 2, 32, 8)
activation = st.sidebar.selectbox(
    "Activation function", ["tanh", "relu", "leaky_relu", "sigmoid"]
)
optimizer = st.sidebar.selectbox(
    "Optimization algorithm", ["adam", "momentum", "sgd"]
)
learning_rate = st.sidebar.slider("Learning rate (η)", 0.005, 1.0, 0.08, 0.005)
epochs = st.sidebar.slider("Training Epochs", 20, 500, 200, 10)

col_btn1, col_btn2 = st.sidebar.columns(2)
train_clicked = col_btn1.button("🚀 Train", use_container_width=True)
verify_clicked = col_btn2.button("🔬 Verify", use_container_width=True)

st.title("🧠 Backpropagation Visualizer")
st.caption(
    "A production-grade neural network trained with **hand-derived backpropagation** "
    "(multivariate chain rule in pure NumPy, zero autograd). Scrub through training to "
    "inspect parameter trajectories, vanishing/exploding gradients, and exact calculus."
)

# ----------------------------- Dataset generation ----------------------------
@st.cache_data
def get_dataset(name, noise_val, seed=42):
    if name == "Moons":
        X, y = make_moons(n_samples=260, noise=noise_val, random_state=seed)
    elif name == "Circles":
        X, y = make_circles(n_samples=260, noise=noise_val, factor=0.45, random_state=seed)
    else:
        X, y = make_blobs(n_samples=260, centers=2, cluster_std=1.0 + noise_val * 3, random_state=seed)
    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y


X, y = get_dataset(dataset_name, noise)

# ----------------------------- Train (on click or init) ----------------------------
if train_clicked or "history" not in st.session_state:
    net = BackpropNetwork(
        input_dim=2,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        activation=activation,
        optimizer=optimizer,
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

# ----------------------------- Verification Feedback ------------------------
if verify_clicked:
    with st.spinner("Computing numerical central differences for all parameters..."):
        passed, max_err, tensor_errs = net.grad_check(X[:25], y[:25])
        if passed:
            st.success(
                f"✅ **Gradient Check PASSED!** Max relative error: `{max_err:.2e}` (Threshold: `< 1e-6`). "
                "Analytical backprop matches finite differences down to machine precision."
            )
            cols = st.columns(4)
            for idx, (p, err) in enumerate(tensor_errs.items()):
                cols[idx].metric(label=f"Relative Error {p}", value=f"{err:.2e}")
        else:
            st.error(f"❌ Gradient Check failed with maximum error: {max_err:.2e}")

# ----------------------------- Tabs ----------------------------
tab_explore, tab_animated, tab_math = st.tabs([
    "📊 Explore Trained Model",
    "🎬 Interactive Suite & Step Debugger",
    "📐 Complete Mathematical Derivations"
])

with tab_explore:
    # ----------------------------- Epoch scrubber ----------------------------
    epoch = st.slider("Scrub to Epoch", 0, n_epochs - 1, n_epochs - 1)

    col1, col2 = st.columns([1.2, 1])

    # ----------------------------- Decision boundary plot ----------------------------
    with col1:
        st.subheader("Decision Boundary Evolution")
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
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.5, linestyles="--")
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="black", s=35)
        ax.set_title(
            f"Epoch {epoch} | Loss: {history['loss'][epoch]:.4f} | Acc: {history['accuracy'][epoch]*100:.1f}%",
            fontsize=12,
            fontweight="bold"
        )
        fig.colorbar(contour, ax=ax, label="P(class = 1)")
        st.pyplot(fig)

    # ----------------------------- Loss & Gradient Norms ----------------------------
    with col2:
        st.subheader("Loss & Accuracy Dynamics")
        fig2, ax2 = plt.subplots(figsize=(5, 2.6))
        ax2.plot(history["loss"], color="#0284c7", label="BCE Loss", linewidth=2)
        ax2.axvline(epoch, color="#f43f5e", linestyle="--", linewidth=1.2)
        ax2.scatter([epoch], [history["loss"][epoch]], color="#f43f5e", zorder=5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.25)
        st.pyplot(fig2)

        st.subheader("Gradient Norms by Layer (Vanishing/Exploding Check)")
        grad_labels = ["W1 (Input→Hidden)", "b1", "W2 (Hidden→Output)", "b2"]
        grad_values = [
            history["grad_W1_norm"][epoch],
            history["grad_b1_norm"][epoch],
            history["grad_W2_norm"][epoch],
            history["grad_b2_norm"][epoch],
        ]
        fig3, ax3 = plt.subplots(figsize=(5, 2.6))
        bars = ax3.barh(grad_labels, grad_values, color=["#38bdf8", "#0284c7", "#a855f7", "#7e22ce"])
        ax3.set_xlabel("‖gradient‖ (L2 norm)")
        ax3.grid(True, alpha=0.25, axis="x")
        st.pyplot(fig3)

    # ----------------------------- Weight heatmaps ----------------------------
    st.subheader("Weight Matrices at This Epoch")
    w_col1, w_col2 = st.columns(2)

    with w_col1:
        fig4, ax4 = plt.subplots(figsize=(5, 2.2))
        im = ax4.imshow(history["W1"][epoch].T, cmap="coolwarm", aspect="auto")
        ax4.set_title("W1 (Input Features → Hidden Units)")
        ax4.set_xlabel("Input Feature (x1, x2)")
        ax4.set_ylabel("Hidden Unit")
        fig4.colorbar(im, ax=ax4)
        st.pyplot(fig4)

    with w_col2:
        fig5, ax5 = plt.subplots(figsize=(5, 2.2))
        im2 = ax5.imshow(history["W2"][epoch], cmap="coolwarm", aspect="auto")
        ax5.set_title("W2 (Hidden Units → Output Unit)")
        ax5.set_xlabel("Output Unit")
        ax5.set_ylabel("Hidden Unit")
        fig5.colorbar(im2, ax=ax5)
        st.pyplot(fig5)

with tab_animated:
    st.subheader("Interactive Web Application Suite")
    st.caption("Embedded live client-side visualizer with 60 FPS decision boundary, dynamic SVG computation graph, and micro-step debugger.")

    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            animated_html = f.read()
        components.html(animated_html, height=1250, scrolling=True)
    else:
        st.error("index.html not found.")

with tab_math:
    st.subheader("Multivariate Calculus Derivations")
    st.markdown(
        r"""
        ### 1. Forward Propagation Graph
        $$\mathbf{Z}_1 = \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1, \quad \mathbf{A}_1 = f(\mathbf{Z}_1)$$
        $$\mathbf{Z}_2 = \mathbf{A}_1\mathbf{W}_2 + \mathbf{b}_2, \quad \hat{\mathbf{y}} = \sigma(\mathbf{Z}_2)$$

        ### 2. Binary Cross-Entropy Loss & Output Cancellation
        $$L(\mathbf{y}, \hat{\mathbf{y}}) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)}\ln \hat{y}^{(i)} + (1 - y^{(i)})\ln(1 - \hat{y}^{(i)}) \right]$$

        Notice how the derivative of the Sigmoid activation cancels out the denominator of the BCE loss:
        $$\frac{\partial L}{\partial \mathbf{Z}_2} = \frac{\partial L}{\partial \hat{\mathbf{y}}} \odot \sigma'(\mathbf{Z}_2) = \frac{\hat{\mathbf{y}} - \mathbf{y}}{\hat{\mathbf{y}}(1 - \hat{\mathbf{y}})} \odot \left[\hat{\mathbf{y}}(1 - \hat{\mathbf{y}})\right] = \frac{1}{N}(\hat{\mathbf{y}} - \mathbf{y}) = \mathbf{\delta}_2$$

        ### 3. Hidden-to-Output Parameter Gradients
        $$\frac{\partial L}{\partial \mathbf{W}_2} = \mathbf{A}_1^T \mathbf{\delta}_2 \in \mathbb{R}^{d_h \times 1}$$
        $$\frac{\partial L}{\partial \mathbf{b}_2} = \sum_{\text{rows}} \mathbf{\delta}_2 \in \mathbb{R}^{1 \times 1}$$

        ### 4. Backpropagating Across Hidden Weights
        $$\frac{\partial L}{\partial \mathbf{A}_1} = \mathbf{\delta}_2 \mathbf{W}_2^T \in \mathbb{R}^{N \times d_h}$$
        $$\mathbf{\delta}_1 = \frac{\partial L}{\partial \mathbf{Z}_1} = \left(\mathbf{\delta}_2 \mathbf{W}_2^T\right) \odot f'(\mathbf{A}_1)$$

        ### 5. Input-to-Hidden Parameter Gradients
        $$\frac{\partial L}{\partial \mathbf{W}_1} = \mathbf{X}^T \mathbf{\delta}_1 \in \mathbb{R}^{d_{in} \times d_h}$$
        $$\frac{\partial L}{\partial \mathbf{b}_1} = \sum_{\text{rows}} \mathbf{\delta}_1 \in \mathbb{R}^{1 \times d_h}$$

        ### 6. Numerical Gradient Verification
        $$\frac{\partial L}{\partial \theta_j} \approx \frac{L(\theta_j + \epsilon) - L(\theta_j - \epsilon)}{2\epsilon}, \quad \text{Relative Error} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|}{\|\nabla_{\text{analytical}}\| + \|\nabla_{\text{numerical}}\|} < 10^{-7}$$
        """
    )
