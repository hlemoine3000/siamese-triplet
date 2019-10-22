import numpy as np
import os
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import datasets, transforms

import evaluation

class MNIST(data.Dataset):
   
    def __init__(self, split, small=False, transform=None):
        # self.transform = transformers.get_basic_transformer()
        self.split = split
        self.transform = transform

        pwd = os.path.dirname(os.path.abspath(__file__))

        if split == "train":

            train_dataset = datasets.MNIST(pwd, train=True, download=True,
                                           transform=None)
            self.X = train_dataset.data.float() / 255.
            self.y = train_dataset.targets

            np.random.seed(2)
            if small:
                ind = np.random.choice(len(train_dataset), 2000, replace=False)
            else:
                ind = np.random.choice(len(train_dataset), len(train_dataset), replace=False)
            self.X = self.X[ind]
            self.y = self.y[ind]

        elif split == "val":

            test_dataset = datasets.MNIST(pwd, train=False, download=True,
                                          transform=None)

            self.X = test_dataset.data.float() / 255.
            self.y = test_dataset.targets

        self.X = self.X.unsqueeze(1)

    def __getitem__(self, index):
        X, y = self.X[index].clone(), self.y[index].clone()

        if self.transform:
            X = self.transform(X)

        return X, y

    def __len__(self):
        """Return size of dataset."""
        return self.X.shape[0]


class SiameseMNIST(data.Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, split, transform=None):

        self.transform = transform
        self.split = split
        pwd = os.path.dirname(os.path.abspath(__file__))

        if split == 'train':

            self.mnist_dataset = datasets.MNIST(pwd, train=True, download=True,
                                           transform=None)

            self.train_labels = self.mnist_dataset.targets
            self.train_data = self.mnist_dataset.data.float() / 255.
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        elif split == "val":

            self.mnist_dataset = datasets.MNIST(pwd, train=False, download=True,
                                          transform=None)

            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.targets
            self.test_data = self.mnist_dataset.data.float() / 255.
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.split == "train":
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]].unsqueeze(0)
            img2 = self.test_data[self.test_pairs[index][1]].unsqueeze(0)
            target = self.test_pairs[index][2]

        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


def get_mnist_trainset(batch_size=50, small=False):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize([0.5], [0.5])])

    mnist_dataset = MNIST(split='train',
                          small=small,
                          transform=pre_process)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True)

    return mnist_data_loader


def get_evaluator(batch_size=50) -> evaluation.Evaluator:
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize([0.5], [0.5])])

    mnist_dataset = SiameseMNIST(split='val',
                                     transform=pre_process)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True)

    return evaluation.Pairs_Evaluator(mnist_data_loader, 'mnist')
