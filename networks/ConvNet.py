import torch
import pandas as pd
import tensorflow as tf
import torchvision
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ConvNet(nn.Module):
  def __init__(self):
    # Call nn.Module's constructor--don't forget this
    super().__init__()

    """
    Define layers
    """
    # Explanation of arguments
    # Remember a Convolution layer will take some input volume HxWxC
    # (H = height, W = width, and C = channels) and map it to some output
    # volume H'xW'xC'.
    #
    # Conv2d expects the following arguments
    #   - C, the number of channels in the input
    #   - C', the number of channels in the output
    #   - The filter size (called a kernel size in the documentation)
    #     Below, we specify 5, so our filters will be of size 5x5.
    #   - The amount of padding (default = 0)
    self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2) #notice how we use padding to prevent dimension reduction
    self.conv2 = nn.Conv2d(10, 15, kernel_size=3, padding=1)

    # Pooling layer takes two arguments
    #   - Filter size (in this case, 2x2)
    #   - Stride
    self.pool = nn.MaxPool2d(10, 2)

    self.fc1 = nn.Linear(800, 400)
    self.fc2 = nn.Linear(400, 100)
    self.fc3 = nn.Linear(100, 25)
    self.fc4 = nn.Linear(25, 5)

  def forward(self, x):
    # Comments below give the shape of x
    # n is batch size

    # (n, 1, 28, 28)
    print("Conv1", x.shape())
    x = self.conv1(x)
    x = F.relu(x)
    # (n, 4, 28, 28)
    x = self.pool(x)
    # (n, 4, 14, 14)
    print("Conv2", x.shape())
    x = self.conv2(x)
    x = F.relu(x)
    # (n, 8, 14, 14)
    x = self.pool(x)
    # (n, 8, 7, 7)
    x = torch.reshape(x, (-1, 800))
    # (n, 8 * 7 * 7)
    x = self.fc1(x)
    x = F.relu(x)
    # (n, 256)
    x = self.fc2(x)
    x = F.relu(x)
    # (n, 128)
    x = self.fc3(x)
    x = F.relu(x)
    # (n, 10)
    x = self.fc4(x)
    return x