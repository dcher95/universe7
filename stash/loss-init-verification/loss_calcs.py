## Examples of loss calculations 
### Useful for verifying initial loss calculations


## Cross-Entropy Loss
import torch.nn as nn
import torch

# Example for 10 classes
n_classes = 10
initial_loss = -torch.log(torch.tensor(1.0 / n_classes))
print(f'Initial Cross-Entropy Loss: {initial_loss.item()}')

## MSE
import torch.nn.functional as F

# Example with zero-initialized predictions and targets
targets = torch.tensor([1.0, 2.0, 3.0])  # Example target values
initial_loss = F.mse_loss(torch.zeros_like(targets), targets)
print(f'Initial MSE Loss: {initial_loss.item()}')


## Huber Loss

# Example with zero-initialized predictions and targets
delta = 1.0  # Huber loss delta parameter
targets = torch.tensor([1.0, 2.0, 3.0])  # Example target values
initial_loss = F.huber_loss(torch.zeros_like(targets), targets, delta=delta)
print(f'Initial Huber Loss: {initial_loss.item()}')

## Binary Cross-Entropy Loss

# Example for binary classification
initial_loss = -torch.log(torch.tensor(0.5))
print(f'Initial Binary Cross-Entropy Loss: {initial_loss.item()}')

## MAE

# Example with zero-initialized predictions and targets
targets = torch.tensor([1.0, 2.0, 3.0])  # Example target values
initial_loss = F.l1_loss(torch.zeros_like(targets), targets)
print(f'Initial MAE Loss: {initial_loss.item()}')
