import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Digit Classification using a Convolutional Neural Network
#                torch.save(model.state_dict(), MODEL_PATH)

# GPU selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load MNIST
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

model_path = "CNN-MNIST"

# Convolutional Neural Network model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1×28×28 → 32×28×28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # → 32×14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32×14×14 → 64×14×14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # → 64×7×7
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x


model = CNN().to(device)


# Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)


# Training loop
for epoch in range(5):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} complete")


# Testing loop
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

cnn = model_path
print(f"Accuracy: {correct / total * 100:.2f}%")