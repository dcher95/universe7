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

# Convert targets to float for Huber Loss
targets = torch.tensor(targets, dtype=torch.float32)

# Load non-pre-trained model and modify for CIFAR-10 (10 classes)
non_pretrained_model = models.resnet18(pretrained=False)
non_pretrained_model.fc = nn.Linear(non_pretrained_model.fc.in_features, 1)  # For regression

# Loss function
delta = 1.0
criterion = nn.SmoothL1Loss()  # Equivalent to Huber Loss with default delta

# Initial Loss Calculation for Non-pretrained Model
non_pretrained_model.eval()
with torch.no_grad():
    outputs = non_pretrained_model(inputs)
    outputs = torch.squeeze(outputs)  # Adjust output shape for loss calculation

# Compute initial loss
initial_loss_non_pretrained = criterion(outputs, targets)
print(f'Initial Huber Loss with Non-pre-trained Weights: {initial_loss_non_pretrained.item()}')

## MSE, MAE and Huber Loss 
# Assuming targets are zero-centered (See below for checking code)
# ideal_loss = torch.mean(torch.tensor([0.0]))  # Placeholder for targets squared mean

# Check if mean is close to zero
# epsilon = 1e-5  # Tolerance for numerical precision
# is_zero_centered = abs(mean.item()) < epsilon