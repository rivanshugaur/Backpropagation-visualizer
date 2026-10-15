# 🧠 Backpropagation Visualizer & Analytical Deep Learning Engine

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20Demo-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://backpropagation-visualizer-blond.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Zero%20Autograd-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Gradient Check](https://img.shields.io/badge/Finite%20Diff-Exact%20(%3C1e--7)-success?style=for-the-badge)](https://en.wikipedia.org/wiki/Numerical_differentiation)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

> **A production-grade neural network trained with hand-derived backpropagation (pure NumPy multivariate calculus, zero autograd dependencies like PyTorch or TensorFlow), accompanied by a real-time 60 FPS interactive web visualizer.**

---

## 🌟 Live Demo on Vercel

Access the live, zero-latency web application deployed globally on Vercel:
👉 **[Launch Interactive Visualizer](https://backpropagation-visualizer-blond.vercel.app)** *(or open `index.html` locally)*

---

## 🎯 Why This Project Stands Out in Technical Interviews

Most machine learning candidates call `loss.backward()` in PyTorch without understanding the underlying linear algebra and multivariate calculus. This project proves fundamental depth by:
1. **Deriving Every Gradient by Hand:** Implementing the full backward pass using pure matrix calculus and the chain rule.
2. **Machine-Precision Verification:** Validating analytical gradients against central finite differences ($f'(\theta) \approx \frac{f(\theta+\epsilon) - f(\theta-\epsilon)}{2\epsilon}$) with a relative error $< 10^{-7}$.
3. **Cancellation Proof Implementation:** Demonstrating why Binary Cross-Entropy loss combined with Sigmoid output cancels the $\sigma'(z)$ term to yield $\delta = \hat{y} - y$, preventing vanishing gradients at the output.
4. **Multiple Optimizers:** From-scratch implementations of SGD, Classical Momentum ($\beta=0.9$), and Adam (with bias-corrected first and second moment vectors).
5. **Architectural Versatility:** Supporting Tanh, ReLU, LeakyReLU (preventing the "Dying ReLU" issue), and Sigmoid.
6. **Zero-Latency Visualization:** A 60 FPS HTML5 Canvas decision boundary coupled with a dynamic SVG computation graph and micro-step backprop debugger.

---

## 📐 Mathematical Foundations

### 1. Forward Propagation Graph
$$\mathbf{Z}_1 = \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1, \quad \mathbf{A}_1 = f(\mathbf{Z}_1)$$
$$\mathbf{Z}_2 = \mathbf{A}_1\mathbf{W}_2 + \mathbf{b}_2, \quad \hat{\mathbf{y}} = \sigma(\mathbf{Z}_2)$$

### 2. Binary Cross-Entropy Loss & Output Cancellation
For $N$ training examples and binary ground truth $\mathbf{y} \in \{0, 1\}^N$:
$$L(\mathbf{y}, \hat{\mathbf{y}}) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)}\ln \hat{y}^{(i)} + (1 - y^{(i)})\ln(1 - \hat{y}^{(i)}) \right]$$

Applying the multivariate chain rule at the output layer:
$$\frac{\partial L}{\partial \mathbf{Z}_2} = \frac{\partial L}{\partial \hat{\mathbf{y}}} \odot \sigma'(\mathbf{Z}_2) = \left[ \frac{\hat{\mathbf{y}} - \mathbf{y}}{\hat{\mathbf{y}}(1 - \hat{\mathbf{y}})} \right] \odot \left[ \hat{\mathbf{y}}(1 - \hat{\mathbf{y}}) \right] = \frac{1}{N}(\hat{\mathbf{y}} - \mathbf{y}) \equiv \mathbf{\delta}_2$$

### 3. Hidden Layer Gradients (Multivariate Chain Rule)
$$\frac{\partial L}{\partial \mathbf{W}_2} = \mathbf{A}_1^T \mathbf{\delta}_2 \in \mathbb{R}^{d_h \times 1}, \quad \frac{\partial L}{\partial \mathbf{b}_2} = \sum_{\text{rows}} \mathbf{\delta}_2 \in \mathbb{R}^{1 \times 1}$$
$$\mathbf{\delta}_1 = \frac{\partial L}{\partial \mathbf{Z}_1} = \left( \mathbf{\delta}_2 \mathbf{W}_2^T \right) \odot f'(\mathbf{Z}_1) \in \mathbb{R}^{N \times d_h}$$
$$\frac{\partial L}{\partial \mathbf{W}_1} = \mathbf{X}^T \mathbf{\delta}_1 \in \mathbb{R}^{d_{in} \times d_h}, \quad \frac{\partial L}{\partial \mathbf{b}_1} = \sum_{\text{rows}} \mathbf{\delta}_1 \in \mathbb{R}^{1 \times d_h}$$

### 4. Finite Difference Numerical Gradient Checking
To mathematically certify that our analytical backprop is bug-free:
$$\nabla_{\text{num}} \theta_i = \frac{L(\theta_i + \epsilon) - L(\theta_i - \epsilon)}{2\epsilon}, \quad \text{Relative Error} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|_2}{\|\nabla_{\text{analytical}}\|_2 + \|\nabla_{\text{numerical}}\|_2} < 10^{-7}$$

---

## 💻 Tech Stack & Architecture

- **Core Engine:** Python 3, NumPy (vectorized array operations, linear algebra, broadcasting).
- **Verification Suite:** Central finite difference numerical checker.
- **Local Interactive Dashboard:** Streamlit, Matplotlib (decision contour rendering, weight heatmaps, loss curves).
- **Web App (Vercel-Ready):** HTML5 Canvas (60 FPS decision boundary), Dynamic SVG (computation graph), Modern CSS (glassmorphism dark UI), KaTeX (LaTeX equation rendering).
- **Deployment:** Vercel Global Edge Network with zero cold-start latency.

---

## 📁 Repository Structure

```
backpropagation-visualizer/
├── index.html         # High-performance interactive web application (Vercel deployment)
├── nn.py              # Pure NumPy neural network engine with manual backprop & grad check
├── app.py             # Feature-rich Streamlit application
├── requirements.txt   # Clean, minimal Python dependencies
├── vercel.json        # Vercel static routing and security headers
└── README.md          # Comprehensive technical documentation & interview guide
```

---

## 🚀 Running Locally

### Option 1: Standalone Web App (No Python Needed)
Simply open `index.html` in any modern web browser or serve via:
```bash
npx serve .
# Or Python's built-in HTTP server:
python3 -m http.server 8000
```

### Option 2: Run Analytical Engine & Unit Tests
```bash
python nn.py
```
*Executes the finite-difference gradient check across all parameter tensors and trains on the Moons dataset with Adam optimizer.*

### Option 3: Streamlit Interactive Dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎓 Key Interview Topics Covered

1. **Why does BCE + Sigmoid simplify to $\hat{y} - y$?** The derivative of the sigmoid function in the numerator cancels out the denominator factors in the BCE loss derivative.
2. **Vanishing Gradient Problem:** The derivative of Sigmoid is bounded by $0.25$, and Tanh by $1.0$. Repeated matrix multiplications over deep networks decay gradients exponentially.
3. **Dying ReLU:** If inputs produce $z \le 0$, the subgradient is $0$, permanently halting updates. Solved by LeakyReLU ($\alpha = 0.01$).
4. **Symmetry Breaking:** Initializing all weights to zero causes hidden units to compute identical activations and receive identical gradients, preventing feature diversification. Solved via Xavier/He initialization.
5. **Gradient Checking:** Verifying analytical Jacobian-vector products using finite differences down to machine precision.

---

## 👤 Author
**Rivanshu Gaur**  
- GitHub: [@rivanshugaur](https://github.com/rivanshugaur)
