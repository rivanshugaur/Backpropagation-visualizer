# Backpropagation Visualizer

An interactive tool that trains a neural network with **backpropagation
implemented entirely from scratch in NumPy** (no PyTorch/TensorFlow autograd)
and lets you scrub through training epoch-by-epoch to see exactly what's
happening under the hood.

## Why this project

Most "I know deep learning" claims are backed by code that calls
`loss.backward()` and trusts an autograd engine to do the work. This project
instead derives and implements every gradient by hand — the chain rule
through each layer is explicit and visible in `nn.py` — to demonstrate real
understanding of how neural networks actually learn.

## What it shows

**Tab 1 — Explore Trained Model**
- **Decision boundary evolution** — watch the network carve up 2D space as
  it trains, epoch by epoch
- **Loss curve** — binary cross-entropy loss over training, with a marker for
  the currently selected epoch
- **Gradient magnitudes per layer** — the actual L2 norm of `dW1`, `db1`,
  `dW2`, `db2` computed by manual backprop at each step
- **Weight matrix heatmaps** — how the learned weights themselves evolve

**Tab 2 — Watch It Learn (Animated)**
- A step-by-step animated walkthrough of one training example: weight
  init → forward pass (values flowing through) → loss calculated →
  backward pass (gradients flowing back, chain rule shown with real
  numbers) → weight update. Auto-play or step manually.
- This is `index.html`, embedded directly into the Streamlit app.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).
Both tabs will be available in the same app — no separate setup needed for
the animated view; `index.html` just needs to stay in the same folder as
`app.py`.

## Project structure

- `nn.py` — the neural network itself: forward pass, manual backpropagation
  (chain rule derived by hand), and gradient descent. Includes a smoke test
  (`python nn.py`) that trains on the moons dataset and asserts the loss
  decreases and accuracy is reasonable.
- `app.py` — the Streamlit UI: dataset selection, hyperparameter controls,
  epoch-by-epoch visualizations, and an embedded tab for the animated demo.
- `index.html` — standalone animated visualization (also works on its own,
  e.g. hosted via GitHub Pages, or embedded inside the Streamlit app).

## Tech stack

Python, NumPy, Matplotlib, Streamlit, scikit-learn (for toy datasets only —
not used anywhere in the network or training loop itself).
