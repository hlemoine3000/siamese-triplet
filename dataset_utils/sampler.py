
import numpy as np

from torch.utils.data import Dataset, Sampler
from torch.utils.data.sampler import BatchSampler
from torchvision import datasets
from random import sample

from dataset_utils import dataset


class Random_BalancedBatchSampler(BatchSampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self,
                 data_source: datasets.DatasetFolder,
                 num_classes_per_batch: int,
                 samples_per_class: int,
                 max_batches: int=1000):

        self.data_source = data_source
        self.num_classes_per_batch = num_classes_per_batch
        self.samples_per_class = samples_per_class
        self.max_batches = max_batches

        if self.num_classes_per_batch > len(self.data_source.classes):
            raise ValueError('Trying to sample {} classes in a dataset with {} classes.'.format(
                self.num_classes_per_batch, len(self.data_source.classes)))

        self.sample_idxs = np.arange(len(self.data_source.samples))
        self.targets = np.array(self.data_source.targets)
        self.classes = np.array(list(self.data_source.class_to_idx.values()))
        self.class_samples = {i: self.sample_idxs[self.targets == i] for i in self.classes}

        self.num_batches = min(len(self.classes) // self.num_classes_per_batch,
                               self.max_batches)

    def __iter__(self):
        batches = []
        shuffled_classes = self.classes.copy()
        np.random.shuffle(shuffled_classes)
        classes_per_batch = []
        for idx in range(0, len(self.classes) - self.num_classes_per_batch, self.num_classes_per_batch):
            classes_per_batch.append(shuffled_classes[idx:idx+self.num_classes_per_batch])
        for classes in classes_per_batch:
            batch = []
            for sample_class in classes:
                if len(self.class_samples[sample_class]) < self.samples_per_class:
                    # raise Exception('Number of samples ({}) is not sufficient in class {}. Require {} samples.'.format(len(self.class_samples[i]), i, self.samples_per_class))
                    batch.append(np.random.choice(self.class_samples[sample_class], self.samples_per_class, replace=True))
                else:
                    batch.append(np.random.choice(self.class_samples[sample_class], self.samples_per_class, replace=False))
            batches.append(np.concatenate(batch))

        return iter(batches)

    def __len__(self):
        return self.num_batches


# class Random_BalancedBatchSampler(BatchSampler):
#     r"""Samples elements sequentially, always in the same order.
#
#     Arguments:
#         data_source (Dataset): dataset to sample from
#     """
#
#     def __init__(self, data_source: datasets.ImageFolder,
#                  num_classes_per_batch: int,
#                  samples_per_class: int,
#                  max_batches: int=1000):
#
#         self.data_source = data_source
#         self.num_classes_per_batch = num_classes_per_batch
#         self.samples_per_class = samples_per_class
#         self.max_batches = max_batches
#
#         if self.num_classes_per_batch > len(self.data_source.classes):
#             raise ValueError('Trying to sample {} classes in a dataset with {} classes.'.format(
#                 self.num_classes_per_batch, len(self.data_source.classes)))
#
#         self.num_batches = len(self.data_source.samples) // (self.num_classes_per_batch * self.samples_per_class)
#
#         self.sample_idxs = np.arange(len(self.data_source.samples))
#         self.targets = np.array(self.data_source.targets)
#         self.classes = np.array(list(self.data_source.class_to_idx.values()))
#         self.class_samples = {i: self.sample_idxs[self.targets == i] for i in self.classes}
#
#     def __iter__(self):
#         batches = []
#         for i in range(min(self.num_batches, self.max_batches)):
#             batch = []
#             chosen_classes_idx = np.random.choice(self.classes, self.num_classes_per_batch, replace=False)
#             for i in chosen_classes_idx:
#                 if len(self.class_samples[i]) < self.samples_per_class:
#                     # raise Exception('Number of samples ({}) is not sufficient in class {}. Require {} samples.'.format(len(self.class_samples[i]), i, self.samples_per_class))
#                     batch.append(np.random.choice(self.class_samples[i], self.samples_per_class, replace=True))
#                 else:
#                     batch.append(np.random.choice(self.class_samples[i], self.samples_per_class, replace=False))
#             batches.append(np.concatenate(batch))
#
#         return iter(batches)
#
#     def __len__(self):
#         return min(self.num_batches, self.max_batches)

class Random_S2VBalancedBatchSampler(BatchSampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: dataset.DatasetS2V,
                 num_classes_per_batch: int,
                 samples_per_class: int,
                 max_batches: int=10000):

        self.data_source = data_source
        self.num_classes_per_batch = num_classes_per_batch
        self.samples_per_class = samples_per_class
        self.max_batches = max_batches

        if self.num_classes_per_batch > len(data_source.classes):
            raise ValueError('Trying to sample {} classes in a dataset with {} classes.'.format(
                self.num_classes_per_batch, len(data_source.classes)))

        self.num_batches = len(self.data_source.samples) // (self.num_classes_per_batch * self.samples_per_class)

        self.sample_idxs = np.arange(len(data_source.samples))
        self.targets = np.array(data_source.targets)
        self.classes = np.array(list(data_source.class_to_idx.values()))
        self.class_samples = {i: self.sample_idxs[self.targets == i] for i in self.classes}
        self.video_only = data_source.video_only
        self.stillclass_to_sampleidx = data_source.stillclass_to_sampleidx

    def __iter__(self):
        batches = []
        for batch_num in range(min(self.num_batches, self.max_batches)):
            batch = []
            chosen_classes_idx = np.random.choice(self.classes, self.num_classes_per_batch, replace=False)
            for i, class_idx in enumerate(chosen_classes_idx):
                if (len(self.class_samples[class_idx]) >= self.samples_per_class):
                    batch.append(np.random.choice(self.class_samples[class_idx], self.samples_per_class, replace=False))
                else:
                    batch.append(np.random.choice(self.class_samples[class_idx], self.samples_per_class, replace=True))
                    # print('No sufficient samples in {} class.'.format(self.data_source.classes[class_idx]))

                if not self.video_only:
                    # Add still image if not in batch
                    if not self.stillclass_to_sampleidx[class_idx] in batch[i]:
                        batch[i][0] = self.stillclass_to_sampleidx[class_idx]

            batches.append(np.concatenate(batch))

        return iter(batches)

    def __len__(self):
        return min(self.num_batches, self.max_batches)


class ImageFolder_BalancedBatchSampler(BatchSampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: datasets.ImageFolder, num_classes: int, samples_per_class: int,
                 num_batches: int = None):
        super().__init__()
        self.data_source = data_source
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.num_batches = num_batches

        if self.num_classes > len(self.data_source.classes):
            raise ValueError('Trying to sample {} classes in a dataset with {} classes.'.format(
                self.num_classes, len(self.data_source.classes)))

    def __iter__(self):
        chosen_classes = sample(self.data_source.classes, self.num_classes)
        chosen_classes_idx = [self.data_source.class_to_idx[c] for c in chosen_classes]

        class_sample_idxs = [[i for i in range(len(self.data_source.samples)) if self.data_source.targets[i] == c] for c
                             in chosen_classes_idx]

        num_samples = min([len(c) for c in class_sample_idxs])
        num_samples -= num_samples % self.samples_per_class
        selected_samples = np.array([sample(c, num_samples) for c in class_sample_idxs])
        batches = np.split(selected_samples, num_samples // self.samples_per_class, axis=1)
        batches = [b.flatten().tolist() for b in batches]
        return iter(batches)

    def __len__(self):
        return len(self.data_source)

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class TrackSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self,
                 data_source: dataset.ImageFolderTrackDataset_with_Labels,
                 samples_per_class: int):
        super().__init__(data_source)

        self.data_source = data_source
        self.samples_per_class = samples_per_class

        self.num_batches = len(self.data_source.cooccurring_tracks)

        self.coocurringtracks_idxs = np.arange(len(self.data_source.cooccurring_tracks))

    def __iter__(self):

        batches = []
        coocurringtracks_idxs_shuffled = self.coocurringtracks_idxs.copy()
        np.random.shuffle(coocurringtracks_idxs_shuffled)
        for i in range(self.num_batches):
            batch = []
            coocurringtracks_idx = coocurringtracks_idxs_shuffled[i]
            for track in self.data_source.cooccurring_tracks[coocurringtracks_idx]:
                samples_indexes = self.data_source.track_idx_to_sample_idx[track]
                num_samples = min(self.samples_per_class, len(samples_indexes))
                chosen_samples_idx = np.random.choice(samples_indexes, num_samples, replace=False)
                batch.append(chosen_samples_idx)
            batches.append(np.concatenate(batch))

        return iter(batches)

    def __len__(self):
        return self.num_batches
