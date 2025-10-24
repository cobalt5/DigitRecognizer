#this file is for figuring out how to use pytorch and numpy to build a multilayerperceptron


#input - a vector of length 784 (28*28) of values between 0 and 1
#output - a vector of length 10 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) of values between 0 and 1

#I will do 2 layers in between, both will be 16 neurons long

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if torch.cuda.is_available():            # NVIDIA (CUDA)
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

