import torch
import numpy as np
import matplotlib.pyplot as plt

# HAND CLALCULATION OF GRADIENTS

# Loss: L = (w·x + b − t)², with w = 1, x = 2, b = 0, t = 10
# Forward: w·x + b − t = 2 − 10 = −8, so L = 64.
# Analytic gradient (chain rule):

# text
# ∂L/∂w = 2(w·x + b − t) · x = 2·(−8)·2 = −32
# ∂L/∂b = 2(w·x + b − t)     = 2·(−8)   = −16

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x, t = 2.0, 10.0

loss = (w * x + b - t) ** 2
loss.backward()

print(f"Autograd: w.grad = {w.grad.item()}, b.grad = {b.grad.item()}")

#activation & loss functions

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def mse(pred, target):
    return np.mean((pred - target) ** 2)

def cross_entropy(probs, targets, eps=1e-12):
    """Mean binary cross-entropy; probs already passed through sigmoid."""
    targets = targets.reshape(probs.shape)
    return -np.mean(targets * np.log(probs + eps)
                    + (1 - targets) * np.log(1 - probs + eps))

print(softmax(np.array([2.0, 1.0, 0.1])))
# [0.659 0.242 0.099]   ✓ sums to 1, preserves ordering, largest logit gets ~66%


# 2-4-1 XOR network (NumPy, manual backprop)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)
N = X.shape[0]

def train(lr, epochs=5000, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, 4)); b1 = np.zeros(4)
    W2 = rng.normal(0, 1, (4, 1)); b2 = np.zeros(1)
    losses = []

    for epoch in range(epochs):
        # ---- forward ----
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)

        loss = cross_entropy(a2, y)
        losses.append(loss)
        if not np.isfinite(loss):          # diverged (large lr)
            print(f"lr={lr}: diverged at epoch {epoch}")
            break

        # ---- backward (BCE + sigmoid ⇒ dL/dz2 simplifies to (a2 - y)) ----
        dz2 = (a2 - y) / N                  # d(mean BCE)/dz2
        dW2 = a1.T @ dz2;  db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)                # ReLU gradient
        dW1 = X.T @ dz1;   db1 = dz1.sum(axis=0)

        # ---- gradient descent update ----
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1

    return losses, sigmoid(relu(X @ W1 + b1) @ W2 + b2)

for lr in [0.01, 0.5, 5.0]:
    losses, preds = train(lr)
    print(f"lr={lr}: final loss = {losses[-1]:.4f}, "
          f"predictions = {np.round(preds.ravel(), 2)}")

plt.figure(figsize=(7, 4))
for lr, style in [(0.01, '--'), (0.5, '-'), (5.0, ':')]:
    losses, _ = train(lr)
    plt.plot(losses[:5000], style, label=f"lr={lr}")
plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('BCE loss (log scale)')
plt.title('XOR training: effect of learning rate'); plt.legend(); plt.show()