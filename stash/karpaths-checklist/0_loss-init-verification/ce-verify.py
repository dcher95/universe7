### Example of calculation using ResNet18 ###
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
criterion = nn.CrossEntropyLoss()

# Initial Loss Calculation for Non-pretrained Model
non_pretrained_model.eval()
with torch.no_grad():
    outputs = non_pretrained_model(inputs)

# Compute initial loss
initial_loss_non_pretrained = criterion(outputs, targets)
print(f'Initial Loss with Non-pre-trained Weights: {initial_loss_non_pretrained.item()}')

# Ideal Loss Calculation
## Cross-Entropy & Binary Cross-Entropy
n_classes = 10
ideal_loss = -torch.log(torch.tensor(1.0 / n_classes))
print(f'Ideal Initial Loss: {ideal_loss.item()}')