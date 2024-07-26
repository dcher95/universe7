import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

# Load a pretrained ResNet18 model
model = resnet18(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Example input: batch of 4 images, each of size 3x224x224
inputs = torch.randn(4, 3, 224, 224)

# Forward pass
outputs = model(inputs)

# Choose the first example in the batch for our debugging
target_index = 0
loss = outputs[target_index].sum()  # Sum of all outputs of example 0

# Backward pass
model.zero_grad()
loss.backward()

# Check the gradients
input_gradients = inputs.grad  # This should be None because we didn't retain the graph
print(f"Gradients of the input:\n{input_gradients}")

# To properly retain the graph for inputs, we need to set requires_grad
inputs.requires_grad = True

# Forward pass again
outputs = model(inputs)
loss = outputs[target_index].sum()

# Backward pass
model.zero_grad()
loss.backward()

# Check the gradients again
input_gradients = inputs.grad  # Now this should have non-zero gradients only for target_index
print(f"Gradients of the input:\n{input_gradients}")

# Verify that only the target_index example has non-zero gradients
non_zero_gradients = (input_gradients[target_index] != 0).any()
other_examples_have_zero_gradients = all((input_gradients[i] == 0).all() for i in range(inputs.size(0)) if i != target_index)

print(f"Non-zero gradients for target index: {non_zero_gradients}")
print(f"Other examples have zero gradients: {other_examples_have_zero_gradients}")
