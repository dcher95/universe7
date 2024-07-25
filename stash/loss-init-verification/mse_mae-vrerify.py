import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# Get a batch of inputs and targets
inputs, targets = next(iter(trainloader))

# Load non-pre-trained model and modify for CIFAR-10 (10 classes)
non_pretrained_model = models.resnet18(pretrained=False)
non_pretrained_model.fc = nn.Linear(non_pretrained_model.fc.in_features, 10)

# Loss function
criterion = nn.MSELoss()  # Or nn.L1Loss() for MAE

# Initial Loss Calculation for Non-pretrained Model
non_pretrained_model.eval()
with torch.no_grad():
    outputs = non_pretrained_model(inputs)

# Compute initial loss
# For MSE and MAE, ensure targets are appropriate (numeric values for regression)
initial_loss_non_pretrained = criterion(outputs, torch.randn_like(outputs))
print(f'Initial MSE Loss with Non-pre-trained Weights: {initial_loss_non_pretrained.item()}')

# Ideal Loss Calculation
# Assuming targets are normalized or zero-centered if required
# This needs to be calculated based on your specific target distribution
ideal_loss = torch.mean(torch.tensor([0.0]))  # Placeholder for targets squared mean or absolute mean
print(f'Ideal Initial MSE Loss: {ideal_loss.item()}')

## MSE, MAE and Huber Loss 
# Assuming targets are zero-centered (See below for checking code)
# ideal_loss = torch.mean(torch.tensor([0.0]))  # Placeholder for targets squared mean

# Check if mean is close to zero
# epsilon = 1e-5  # Tolerance for numerical precision
# is_zero_centered = abs(mean.item()) < epsilon