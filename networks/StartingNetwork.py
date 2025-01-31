import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(600 * 800 * 3, 5)
        self.sigmoid = nn.Sigmoid()
        # check output dimensions
        
    def forward(self, x):
        x = self.flatten(x)
        x = x.to(torch.float32)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
