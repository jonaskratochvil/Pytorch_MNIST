#!/usr/bin/env python3

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from MNIST_loader import MNIST

mnist = MNIST()


class Net(nn.Module):
    # This defines the inheritance from parent class for Net
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)  # kernel width is 3
        # Here goes out 16 filters of size 6 x 6 -> for flattening do 6*6*16
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)  # inputs, outputs
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Do max pooling over the 2x2 region (if max pooling operation is square you can just write 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # take everything except the batch size
        num_features = 1
        # Iterate over all dimmensions and multiply them to get to one dimmension
        # e.g. if I have 16 filters of size  6*6 -> 16*6*6
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
