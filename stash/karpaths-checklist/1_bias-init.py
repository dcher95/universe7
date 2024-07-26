## Initialize the final layer weights correctly.

## Regression Example
import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the final layer for regression (e.g., predicting a single value)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

# Initialize final layer weights and bias
nn.init.normal_(model.fc.weight, mean=0, std=0.01)
model.fc.bias.data.fill_(50)  # Assume mean of target values is 50

# Print model to verify changes
print(model)

## Binary Classification Example
import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the final layer for binary classification (e.g., output a single logit)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

# Initialize final layer weights and bias
nn.init.normal_(model.fc.weight, mean=0, std=0.01)

# For 1:10 ratio of positives:negatives, set bias to log(0.1 / 0.9)
model.fc.bias.data.fill_(torch.log(torch.tensor(0.1 / 0.9)))

# Print model to verify changes
print(model)


## Multi-class Classification Example
import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the final layer for multi-class classification (e.g., 3 classes)
num_features = model.fc.in_features
num_classes = 3
model.fc = nn.Linear(num_features, num_classes)

# Initialize final layer weights
nn.init.normal_(model.fc.weight, mean=0, std=0.01)

# Assuming class ratios are 1:2:7
class_ratios = torch.tensor([1/10, 2/10, 7/10])  # Class ratios sum to 1
model.fc.bias.data = torch.log(class_ratios)  # Logits initialization

# Print model to verify changes
print(model)
