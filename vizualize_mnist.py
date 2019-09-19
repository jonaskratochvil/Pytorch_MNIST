#!/usr/bin/env python3

import matplotlib.pyplot as plt
from MNIST_loader import MNIST

# Vizualize first number from MNIST dataset

mnist = MNIST()

label = mnist.train.data["labels"][1]

plt.title(f"The label is {label}")
plt.imshow(mnist.train.data["images"][1].reshape((28, 28)), cmap="gray")
plt.show()
