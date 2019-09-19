import sys
import os
import numpy as np


class MNIST:
    # here we define the dataset parameters
    H, W, C = 28, 28, 1  # height, width, channel
    labels = 10

    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            # Denote the internal variables by underscore at the beginning
            self._data = data
            # make images numpy floats and normalize them
            self._data["images"] = self._data["images"].astype(np.float32)/255
            # find how many images we have in our MNIST dataset
            self._size = len(self._data["images"])
            # check if we want to shuffle
            self._shuffler = np.random.RandomState(
                seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None):
            # If we want to permute to it else just arrange
            permutation = self._shuffler.permutation(
                self._size) if self._shuffler else np.arrange(self._size)

            while len(permutation):
                # Here if size is not provided -> size is None and therefore inf
                # is chosem which will never be min to it will just take whole
                # dataset as one big batch
                # This also allows us to deal with cases when batch size is larger
                # than the rest of the dataset
                batch_size = min(size or np.inf, len(permutation))
                # batch perm takes the batch size chunk and than we chop the
                # permutation for this chunk
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                # data are dictionary with keys images and labels and we save
                # it for each batch chunk
                batch = {}
                for key in self._data:
                    batch[key] = self._data[key][batch_perm]
                yield batch

    def __init__(self, dataset="mnist"):
        path = f"{dataset}.npz"
        if not os.path.exists(path):
            raise ValueError(
                "You must first download a MNIST dataset to run this code")

        mnist = np.load(path)

        for dataset in ["train", "dev", "test"]:
            data = dict((key[len(dataset) + 1:], mnist[key])
                        for key in mnist if key.startswith(dataset))
            # This gives each of train, dev, test properties of Dataset class, which in turn
            # enables us to do things like mnist.data.train["images"], this also shuffles only
            # for train data
            setattr(self, dataset, self.Dataset(
                data, shuffle_batches=dataset == "train"))
