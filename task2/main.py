import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and scale breast cancer dataset

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features (fit only on training data)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

# Create CsvDataset class

class CsvDataset(Dataset):

    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


train_dataset = CsvDataset(X_train, y_train)
val_dataset = CsvDataset(X_val, y_val)

# Create DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# Build MLP model

class MLP(nn.Module):

    def __init__(self, input_size):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


model = MLP(input_size=X_train.shape[1])

# Print parameter count
total_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Params:", total_params)

# Loss function and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training loop with early stopping

epochs = 40
patience = 7

best_val_accuracy = 0.0
best_epoch = 0
counter = 0

train_losses = []

best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(epochs):

    # Training phase
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)

    # Validation phase
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    # Early stopping
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        counter = 0

    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered!")
        break


# Restore best model
model.load_state_dict(best_model_weights)

print("\nBest val accuracy:", round(best_val_accuracy, 4))
print("Best epoch:", best_epoch)

# Save model state_dict

torch.save(
    model.state_dict(),
    "breast_cancer_model.pth"
)

print("\nModel saved successfully!")

# Load state_dict into a new model

new_model = MLP(input_size=X_train.shape[1])

new_model.load_state_dict(
    torch.load(
        "breast_cancer_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

new_model.eval()

# Evaluate original and reloaded models

def evaluate(model, loader):

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for inputs, labels in loader:

            outputs = model(inputs)

            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.numpy())
            actual.extend(labels.numpy())

    return accuracy_score(actual, predictions)


original_accuracy = evaluate(model, val_loader)
reloaded_accuracy = evaluate(new_model, val_loader)

print("Original model accuracy:", original_accuracy)
print("Reloaded model accuracy:", reloaded_accuracy)

print(
    "Identical:",
    original_accuracy == reloaded_accuracy
)

# Compare with Logistic Regression

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

lr_predictions = logistic_model.predict(X_val)

lr_accuracy = accuracy_score(
    y_val,
    lr_predictions
)

print("\nLogistic Regression Accuracy:", lr_accuracy)

print("PyTorch MLP Accuracy:", original_accuracy)
