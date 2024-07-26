import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define transformations for the training set, flip the images randomly, crop, and normalize
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Load the training set with torchvision
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Show images
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % trainset.classes[labels[j]] for j in range(4)))

# Load pre-trained ResNet18 model
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Modify the final layer for CIFAR-10

# Forward pass through the network (for demonstration)
outputs = model(images)

# Visualize data just before feeding into the model
def visualize_data(images, labels, model):
    imshow(torchvision.utils.make_grid(images))
    print('Labels:', ' '.join('%5s' % trainset.classes[labels[j]] for j in range(4)))
    y_hat = model(images)
    return y_hat

# Visualize the data and get the network output
y_hat = visualize_data(images, labels, model)
