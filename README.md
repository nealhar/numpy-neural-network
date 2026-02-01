# Neural Network From Scratch (NumPy)

This project is a from-scratch implementation of a fully connected neural network using only NumPy. The goal was to deeply understand the internal mechanics of modern machine learning systems by manually building every component: forward propagation, backpropagation, loss functions, and optimization.

Rather than relying on frameworks such as PyTorch or TensorFlow, this project focuses on the mathematical and algorithmic foundations behind neural networks.

The model is trained on a nonlinear spiral classification dataset to demonstrate how multilayer networks learn complex decision boundaries.

---

## What I Built

- Multi-layer neural network (2 → 256 → 256 → 3)
- Dense (fully connected) layers with ReLU activations
- Softmax output layer for multiclass classification
- Categorical Cross-Entropy loss
- Combined Softmax + Cross-Entropy backward pass for numerical stability
- Manual backpropagation through all layers
- Stochastic Gradient Descent optimizer with:
  - Momentum
  - Learning rate decay
- Custom spiral dataset generator
- Dataset visualization using Matplotlib

No external machine learning frameworks were used.

---

## Dataset Visualization

Two images are included to illustrate the training data:

- `spiral_dataset.png` shows the raw spiral dataset, with all points plotted in a single color to highlight the overall geometric structure of the data.

- `colored_spiral_dataset.png` shows the same dataset, but with points grouped by color according to class labels. This visualization highlights how the three classes form interleaving spiral arms and demonstrates why the problem is not linearly separable.

These visualizations help illustrate the nonlinear nature of the classification task and motivate the use of a multi-layer neural network.

---
## What I Learned

### Neural Network Fundamentals
- How forward propagation transforms inputs through linear layers and nonlinear activations
- How gradients flow backward through a network using the chain rule
- Why ReLU introduces nonlinearity and enables deep models
- How Softmax converts logits into probabilities
- Why cross-entropy is used for classification tasks

### Optimization & Training
- Implemented SGD from scratch and extended it with momentum
- Observed how momentum accelerates convergence and reduces oscillation
- Implemented learning rate decay to stabilize late-stage training
- Gained intuition for how learning rate, depth, and width affect convergence

### Numerical Stability
- Applied softmax stabilization using max subtraction
- Used probability clipping to prevent log(0)
- Implemented the combined Softmax + Cross-Entropy gradient simplification

### Practical Engineering Skills
- Designed modular layer abstractions (Dense, ReLU, Loss, Optimizer)
- Debugged exploding and vanishing gradients
- Tuned hyperparameters to improve convergence on nonlinear data
- Visualized datasets and training behavior
- Structured code for clarity and extensibility

---

## Results

After training, the network achieves approximately **98%+ accuracy** on the spiral dataset, successfully learning nonlinear class boundaries.

---

## Technologies Used

- Python
- NumPy
- Matplotlib

---

## Why This Project

I built this project to move beyond “black-box” machine learning and gain hands-on experience with:

- Gradient-based optimization
- Neural network internals
- Numerical stability techniques
- Systems-level thinking in ML pipelines

This experience strengthened both my mathematical understanding and my software engineering skills.

