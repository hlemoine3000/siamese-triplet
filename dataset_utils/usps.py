"""Dataset setting and data loader for USPS.

Modified from
https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/dataset_usps.py
"""

import gzip
import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

import evaluation


class USPS(data.Dataset):
    def __init__(self, root, split, transform=None, download=False, small=False):
        """Init USPS dataset."""
        # init params
        self.split=split
        self.root = os.path.expanduser(root)

        self.filename = "usps_28x28.pkl"

        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None
        
        self.X, self.y = self.load_samples()
        if self.split=="train":
            total_num_samples = self.y.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.X = self.X[indices[0:self.dataset_size], ::]
            self.y = self.y[indices[0:self.dataset_size]]
            
            n_usps = self.y.shape[0]
            np.random.seed(1)
            if small:
                ind = np.random.choice(n_usps, 1800, replace=False)
            else:
                ind = np.random.choice(n_usps, n_usps, replace=False)

            self.X = self.X[ind]
            self.y = self.y[ind]

            self.dataset_size = self.y.shape[0]
        

        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __getitem__(self, index):
        img, label = self.X[index].clone(), self.y[index].clone()
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        return self.dataset_size

    def load_samples(self):
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.split == "train":
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        elif self.split == "val":            
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


class SiameseUSPS(data.Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, root, split, transform=None, download=False, small=False):

        """Init USPS dataset."""
        # init params
        self.split = split
        self.root = os.path.expanduser(root)

        self.filename = "usps_28x28.pkl"

        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        self.X, self.y = self.load_samples()
        if self.split == "train":
            total_num_samples = self.y.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.X = self.X[indices[0:self.dataset_size], ::]
            self.y = self.y[indices[0:self.dataset_size]]

            n_usps = self.y.shape[0]
            np.random.seed(1)
            if small:
                ind = np.random.choice(n_usps, 1800, replace=False)
            else:
                ind = np.random.choice(n_usps, n_usps, replace=False)

            self.X = self.X[ind]
            self.y = self.y[ind]

            self.dataset_size = self.y.shape[0]

        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

        self.labels_set = set(self.y.numpy())
        self.label_to_indices = {label: np.where(self.y.numpy() == label)[0]
                                 for label in self.labels_set}

        if split == "val":

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.y[i].item()]),
                               1]
                              for i in range(0, len(self.X), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.y[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.X), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.split == "train":
            target = np.random.randint(0, 2)
            img1, label1 = self.X[index], self.y[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.X[siamese_index]
        else:
            img1 = self.X[self.test_pairs[index][0]]
            img2 = self.X[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return self.dataset_size

    def load_samples(self):
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.split == "train":
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        elif self.split == "val":
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


def get_usps_trainset(batch_size: int=50,
                      small: bool=False):
    """Get USPS dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize([0.5], [0.5])])

    # dataset and data loader
    usps_dataset = USPS(root="datasets",
                        split='train',
                        transform=pre_process,
                        small=small)

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    return usps_data_loader


def get_evaluator(batch_size=50):
    """Get USPS dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize([0.5], [0.5])])

    # dataset and data loader
    usps_dataset = SiameseUSPS(root="datasets",
                               split='val',
                               transform=pre_process)

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    return evaluation.Pairs_Evaluator(usps_data_loader, 'ups')
