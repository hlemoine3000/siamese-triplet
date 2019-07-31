
import os
import numpy as np
import tqdm
import pickle

from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import default_loader, has_file_allowed_extension

import utils

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


class NumpyDataset(data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)


class ImageFolderTrackDataset(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

            root/class_x/xxx.ext
            root/class_x/xxy.ext
            root/class_x/xxz.ext

            root/class_y/123.ext
            root/class_y/nsdf3.ext
            root/class_y/asd932_.ext

        Args:
            root (string): Root directory path.
            transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            target_transform (callable, optional): A function/transform that takes
                in the target and transforms it.

         Attributes:
            classes (list): List of the class names.
            class_to_idx (dict): Dict with items (class_name, class_index).
            samples (list): List of (sample path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
        """

    def __init__(self, root, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        samples = self._make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.track_targets = [s[1] for s in samples]
        self.gt_targets = [s[2] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _make_dataset(self, dir, class_to_idx, extensions):
        images = []
        gtclass_to_idx = {}
        gt_idx = 0
        dir = os.path.expanduser(dir)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        track_name, gtlabel_name = target.split('_')
                        track_idx = int(''.join(filter(str.isdigit, track_name)))
                        if gtlabel_name not in gtclass_to_idx.keys():
                            gtclass_to_idx[gtlabel_name] = gt_idx
                            gt_idx += 1
                        item = (path, track_idx, gtclass_to_idx[gtlabel_name])
                        images.append(item)

        return images

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, track_target, gt_target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            track_target = self.target_transform(track_target)
        if self.target_transform is not None:
            gt_target = self.target_transform(gt_target)

        return sample, track_target, gt_target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class TrackDataset(data.Dataset):
    def __init__(self,
                 pkl_file,
                 transform=None):

        self.pkl_file = pkl_file
        self.transform = transform

        # Read pickle file
        file2 = open(pkl_file, 'rb')
        data = pickle.load(file2)
        file2.close()

        self.cropped_image_list = data['cropped_images']
        self.track_id = data['track_id']
        self.cooccurring_tracks = data['cooccurring_tracks']
        self.track_to_bbxidx = data['track_to_bbxidx']

    def __getitem__(self, index):
        sample = self.cropped_image_list[index]
        target = self.track_id[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.cropped_image_list)


class PairsDataset(data.Dataset):

    def __init__(self,
                 data_source: str,
                 pair_file: str,
                 transform: transforms,
                 preload: bool=False):
        self.data_source = data_source
        self.pair_file = pair_file
        self.transform = transform

        self.pairs, self.issame = utils.Get_Pairs(pair_file, data_source)

        self.preloaded = False
        if preload:
            print('Preload images')
            self.images = {}
            uniques = np.unique(np.array(self.pairs))
            tbar = tqdm.tqdm(uniques)
            for path in tbar:
                img = Image.open(path)
                self.images[path] = img.copy()
            self.preloaded = True

            # def load_image(fname):
            #     image = Image.open(fname)
            #     return (fname, image.copy())
            # uniques = np.unique(np.array(self.pairs))
            # with mp.Pool(4) as p:
            #     self.images = p.map(load_image, uniques)
            # self.images = {k:v for k,v in self.images}
            # self.preloaded = True


    def __len__(self):
        return len(self.issame)

    def __getitem__(self, idx):
        if self.preloaded:
            img1 = self.images[self.pairs[idx][0]]
            img2 = self.images[self.pairs[idx][1]]
        else:
            img1 = default_loader(self.pairs[idx][0])
            img2 = default_loader(self.pairs[idx][1])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return [img1, img2], self.issame[idx]


class PairsDatasetS2V(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 still_source: str,
                 video_source: str,
                 pair_file: str,
                 transform: transforms,
                 fold_list: list,
                 num_folds: int=10,
                 preload: bool=False):



        self.still_source = still_source
        self.video_source = video_source
        self.pair_file = pair_file
        self.transform = transform

        self.subject_list, self.nb_folds = utils.get_subject_list(pair_file)
        self.nb_subject = len(self.subject_list)
        self.nb_subject_per_fold = self.nb_subject // self.nb_folds
        if num_folds != self.nb_folds:
            raise Exception('There are {} folds in pair file. Marked {} folds.'.format(self.nb_folds, num_folds))
        if max(fold_list) > self.nb_folds:
            raise Exception('Fold number {} is out of range. Maximum number of fold is {}.'.format(max(fold_list), self.nb_folds))

        self.pairs, self.issame = utils.get_pairs_from_fold(still_source,
                                                            video_source,
                                                            pair_file,
                                                            self.subject_list)

        self.preloaded = False
        if preload:
            self.images = {}
            uniques = np.unique(np.array(self.pairs))
            for path in uniques:
                img = Image.open(path)
                self.images[path] = img.copy()
            self.preloaded = True

            # def load_image(fname):
            #     image = Image.open(fname)
            #     return (fname, image.copy())
            # uniques = np.unique(np.array(self.pairs))
            # with mp.Pool(4) as p:
            #     self.images = p.map(load_image, uniques)
            # self.images = {k:v for k,v in self.images}
            # self.preloaded = True

    def __len__(self):
        return len(self.issame)

    def __getitem__(self, idx):
        if self.preloaded:
            img1 = self.images[self.pairs[idx][0]]
            img2 = self.images[self.pairs[idx][1]]
        else:
            img1 = default_loader(self.pairs[idx][0])
            img2 = default_loader(self.pairs[idx][1])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return [img1, img2], self.issame[idx]


class DatasetS2V(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 still_source: str,
                 video_source: str,
                 pair_file: str,
                 transform: transforms,
                 fold_list: list,
                 num_folds: int=10,
                 preload: bool = False,
                 video_only: bool = False,
                 samples_division_list = None,  # [0.4, 0.6]
                 div_idx: int = -1):

        self.still_source = still_source
        self.video_source = video_source
        self.pair_file = pair_file
        self.transform = transform
        self.video_only = video_only

        self.subject_list, self.nb_folds = utils.get_subject_list(pair_file)
        self.nb_subject = len(self.subject_list)
        self.nb_subject_per_fold = self.nb_subject // self.nb_folds
        if num_folds != self.nb_folds:
            raise Exception('There are {} folds in pair file. Marked {} folds.'.format(self.nb_folds, num_folds))
        if max(fold_list) > self.nb_folds:
            raise Exception('Fold number {} is out of range. Maximum number of fold is {}.'.format(max(fold_list), self.nb_folds))

        # Divide samples within subject subdirectories
        if samples_division_list:
            if sum(samples_division_list) != 1.0:
                raise Exception('The sample division list is incorrect as the division sum should be 1. The sum is {}.'.format(sum(samples_division_list)))
            if (div_idx >= len(samples_division_list)) or (div_idx < 0):
                raise Exception('The division index ({}) must be a valid index for the division list {}.'.format(div_idx, samples_division_list))

        # Transform class name to index
        subject_set = utils.extract_fold_list(fold_list, self.subject_list, self.nb_subject_per_fold)
        class_to_idx = {}
        for class_idx, subject in enumerate(subject_set):
            class_to_idx[subject] = class_idx

        samples = []
        stillclass_to_sampleidx = {}
        tbar = tqdm.tqdm(subject_set)
        for i, subject in enumerate(tbar):
            subject_video_path = os.path.join(self.video_source, subject)
            video_image_paths = utils.get_image_paths(subject_video_path)

            # Divide video samples if requested
            if samples_division_list:
                lower_div_idx = int(len(video_image_paths) * sum(samples_division_list[:div_idx]))
                upper_div_idx = int(len(video_image_paths) * sum(samples_division_list[:div_idx + 1]))
                video_image_paths = video_image_paths[lower_div_idx:upper_div_idx]

            still_image_path = os.path.join(self.still_source, subject + '_0000.JPG')
            if not os.path.isfile(still_image_path):
                raise Exception('Still image not found at {}'.format(still_image_path))

            if video_only:
                paths = video_image_paths
            else:
                paths = [still_image_path] + video_image_paths
                # (class_idx, sample_idx)
                stillclass_to_sampleidx[class_to_idx[subject]] = len(samples)

            for path in paths:
                item = (path, class_to_idx[subject])
                samples.append(item)

        self.stillclass_to_sampleidx = stillclass_to_sampleidx

        self.classes = subject_set
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.preloaded = False
        if preload:
            print('Preloading images...')
            self.images = {}
            tbar = tqdm.tqdm(self.samples)
            for path, lbl in tbar:
                img = Image.open(path)
                self.images[path] = img.copy()
            self.preloaded = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        """
                Args:
                    index (int): Index

                Returns:
                    tuple: (sample, target) where target is class_index of the target class.
                """

        path, label = self.samples[idx]
        if self.preloaded:
            sample = self.images[path]
        else:
            sample = default_loader(path)

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class DatasetS2V_from_subject(data.Dataset):

    def __init__(self,
                 still_dir,
                 video_dir,
                 video_list,
                 subject_list,
                 transform: transforms,
                 max_samples_per_subject=10,
                 video_only=False):

        self.transform = transform

        image_path_list = []
        labels_list = []
        samples = []

        for video_view in video_list:
            for subject in subject_list:

                subject_video_path = os.path.join(video_dir, video_view, subject)
                video_image_paths = utils.get_image_paths(subject_video_path)
                if len(video_image_paths) > max_samples_per_subject:
                    video_image_paths = video_image_paths[0:max_samples_per_subject]

                label = subject + '_' + video_view
                labels_list += [label] * len(video_image_paths)

                if not video_only:
                    label = subject + '_still'
                    if label not in labels_list: # No need to add still image a second time.
                        video_image_paths.append(os.path.join(still_dir, subject + '_0000.JPG'))
                        labels_list.append(label)

                image_path_list += video_image_paths

        classes = utils.unique(labels_list)
        class_to_idx = {key: tgt for (tgt, key) in enumerate(classes)}

        for i, image_path in enumerate(image_path_list):

            item = (image_path, class_to_idx[labels_list[i]])
            samples.append(item)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        """
                Args:
                    index (int): Index

                Returns:
                    tuple: (sample, target) where target is class_index of the target class.
                """

        path, label = self.samples[idx]
        sample = default_loader(path)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
