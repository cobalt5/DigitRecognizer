#RESTART THIS LATER

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

train_val_images = 'train_images.npy' # Train 80%, Validation 20%
train_val_labels = 'train_labels.npy' # Train 80%, Validation 20%
test_images = 'test_images.npy'
test_labels = 'test_labels.npy'

train_val_images = np.load(train_val_images) # shape (60000, 28, 28)
train_val_labels = np.load(train_val_labels) # shape (60000,)

# 90% of the training data for training, 10% for validation
train_images = train_val_images[:int(train_val_images.shape[0] * 0.9)]
train_labels = train_val_labels[:int(train_val_labels.shape[0] * 0.9)]

val_images = train_val_images[int(train_val_images.shape[0] * 0.9):]
val_labels = train_val_labels[int(train_val_labels.shape[0] * 0.9):]

test_images = np.load(test_images) # shape (10000, 28, 28)
test_labels = np.load(test_labels) # shape (10000,)

#we need to divide the images by 255 to get them between 0 and 1
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# Check CUDA availability and set device
device = "cpu"  # default to CPU
if torch.cuda.is_available():
    try:
        test_tensor = torch.zeros(1).cuda()  # Test CUDA functionality
        device = "cuda"
        torch.cuda.empty_cache()  # Clear any cached memory
    except RuntimeError as e:
        print(f"CUDA error: {e}")
        print("Falling back to CPU")
        device = "cpu"

print(f"Using {device} device")


#convert to torch tensors
#CrossEntropyLoss in PyTorch expects the labels to be of type Long
train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_images = torch.tensor(val_images, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Create datasets with CPU tensors
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
#run the training loop from here
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")