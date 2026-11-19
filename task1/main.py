import torch
import numpy as np

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