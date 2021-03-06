
import os
import numpy as np
from collections import OrderedDict
from PIL import Image
import tqdm
import ntpath


class FoldGenerator():
    def __init__(self, num_fold: int, num_train: int, num_val: int):
        self.fold_list = list(range(num_fold))
        self.train_idx = num_train
        self.val_idx = num_train + num_val

    def get_fold(self):
        return (self.fold_list[:self.train_idx], self.fold_list[self.train_idx:self.val_idx], self.fold_list[self.val_idx:])

    def permute(self):
        self.fold_list.append(self.fold_list.pop(0))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.values = []
        self.counter = 0

    def append(self, val) -> None:
        self.values.append(val)
        self.counter += 1

    @property
    def val(self) -> float:
        return self.values[-1]

    @property
    def avg(self) -> float:
        if len(self.values) != 0:
            return sum(self.values) / len(self.values)
        else:
            return 0

    def last_avg(self) -> float:
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class AverageData_Dict(object):

    def __init__(self):
        self.data_dict = {}

    @property
    def get_avg(self) -> dict:
        return {k:d.avg for k,d in self.data_dict.items()}

    def get_last_avg(self) -> dict:
        return {k:d.last_avg() for k,d in self.data_dict.items()}

    def __getitem__(self, attr: str) -> AverageMeter:
        if attr not in self.data_dict.keys():
            self.data_dict[attr] = AverageMeter()
        return self.data_dict[attr]


class AttrDict(dict):
    """ Nested Attribute Dictionary

    A class to convert a nested Dictionary into an object with key-values
    accessibly using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse down nested dicts (like: AttrDict.attr.attr)
    """

    def __init__(self, mapping=None):
        super(AttrDict, self).__init__()
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value  # for code completion in editors

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    __setattr__ = __setitem__


def gaussian(x, mu, sig):
    return 1 / (sig * np.sqrt(2*np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sig ** 2))


def make_square(im: Image, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new(im.mode, (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def state_dict_to_cpu(state_dict: OrderedDict):
    new_state = OrderedDict()
    for k in state_dict.keys():
        new_state[k] = state_dict[k].cpu()
    return new_state


def get_image_paths(image_dir):
    image_paths = []
    if os.path.isdir(image_dir):
        images = os.listdir(image_dir)
        image_paths = [os.path.join(image_dir,img) for img in images]
    return image_paths

def extract_fold_list(fold_list,
                      subject_list,
                      nb_subject_per_fold):

    list = []
    for fold in fold_list:
        upper_idx = fold * nb_subject_per_fold + nb_subject_per_fold
        lower_idx = fold * nb_subject_per_fold
        list += subject_list[lower_idx: upper_idx]

    return list


def get_subject_list(pairs_path):

    subject_list = []

    with open(pairs_path, 'r') as f:

        nb_fold = f.readline().split('\t')[0]

        for line in f.readlines()[1:]:
            pair = line.strip().split()
            subject_list.append(pair[0])

        subject_list = unique(subject_list)

    return subject_list, int(nb_fold)


def get_paths_from_file(subject_filename,
                        still_path,
                        video_path,
                        max_subject=10,
                        max_images_per_subject=10,
                        tag=''):
    path_list = []
    label_list = []

    subjects_list = []
    with open(subject_filename, 'r') as f:
        for line in f.readlines()[1:]:
            subjects_list.append(line.strip())

    num_subject = 0
    for subject in subjects_list:

        # Get still image
        still_image_path = os.path.join(still_path, subject + '_0000.JPG')
        path_list.append(still_image_path)
        label_list.append(subject + '_still' + tag)

        video_subject_dir = os.path.join(video_path, subject)
        subject_images_list = os.listdir(video_subject_dir)

        images_per_subject = 0
        for subject_image in subject_images_list:
            path = os.path.join(video_subject_dir, subject_image)

            if os.path.exists(path):
                path_list.append(path)
                label_list.append(subject + tag)
                images_per_subject += 1

            if images_per_subject >= max_images_per_subject:
                break

        num_subject += 1
        if num_subject >= max_subject:
            break

    return path_list, label_list


def get_pairs_from_fold(still_source,
                        video_source,
                        pair_file,
                        fold_subject_list):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []

    pairs = []
    with open(pair_file, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)

    tbar = tqdm.tqdm(pairs)
    for pair in tbar:
        if pair[0] in fold_subject_list:
            if len(pair) == 3:

                path0 = add_extension(
                    os.path.join(still_source, pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(
                    os.path.join(video_source, pair[0], pair[0] + '_' + '%d' % int(pair[2])))
                issame = True

                if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                    path_list.append([path0, path1])
                    issame_list.append(issame)
                else:
                    nrof_skipped_pairs += 1

            elif len(pair) == 4:

                path0 = add_extension(
                    os.path.join(still_source, pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(
                    os.path.join(video_source, pair[2], pair[2] + '_' + '%d' % int(pair[3])))
                issame = False

                if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                    path_list.append([path0, path1])
                    issame_list.append(issame)
                else:
                    nrof_skipped_pairs += 1

    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    if os.path.exists(path + '.JPG'):
        return path + '.JPG'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


# Python3 program to Convert a
# list to dictionary
# https://www.geeksforgeeks.org/python-convert-a-list-to-dictionary/
def list_to_dict(lst) -> dict:
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct


def read_file_to_list(filename: str):
    with open(filename) as file:
        file_list = [[int(n) if n.rstrip("\n").isnumeric() else n for n in line.split(',')] for line in file]
    return file_list


def write_list_to_file(filename: str,
                       list_to_write: list):
    with open(filename, 'w') as file:
        file.writelines(','.join(str(j) for j in i) + '\n' for i in list_to_write)


def read_pairs(pairs_filename: str):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


# Set up for evaluation
def get_pairs(pairs_path,
              images_path):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []

    print('Reading pairs at {}'.format(pairs_path))
    pairs = read_pairs(pairs_path)

    path0 = ''
    path1 = ''
    issame = False

    tbar = tqdm.tqdm(pairs)
    for pair in tbar:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(images_path, pair[0], pair[1]))
            path1 = add_extension(os.path.join(images_path, pair[0], pair[2]))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(images_path, pair[0], pair[1]))
            path1 = add_extension(os.path.join(images_path, pair[2], pair[3]))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list.append([path0, path1])
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            # print list
    return unique_list


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
