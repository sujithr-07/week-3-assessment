# HAND CLALCULATION OF GRADIENTS


# Loss: L = (w·x + b − t)², with w = 1, x = 2, b = 0, t = 10

# Forward: w·x + b − t = 2 − 10 = −8, so L = 64.

# Analytic gradient (chain rule):

# text
# ∂L/∂w = 2(w·x + b − t) · x = 2·(−8)·2 = −32
# ∂L/∂b = 2(w·x + b − t)     = 2·(−8)   = −16


import torch

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x, t = 2.0, 10.0

loss = (w * x + b - t) ** 2
loss.backward()

print(f"Autograd: w.grad = {w.grad.item()}, b.grad = {b.grad.item()}")
