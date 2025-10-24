#This file is for figuring out how to import the MNIST digits dataset, and doing the prepreocessing to get it ready for training a neural network

import numpy as np
import torch

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

#convert to torch tensors
#CrossEntropyLoss in PyTorch expects the labels to be of type Long
train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_images = torch.tensor(val_images, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

