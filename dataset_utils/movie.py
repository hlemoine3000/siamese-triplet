
import torch
from torch.utils import data

from dataset_utils.dataset import ImageFolderTrackDataset
from dataset_utils.sampler import TrackSampler

def get_trainset(test_dir,
                 samples_per_class: int,
                 transform=None):

    print('TRAIN SET BBT.')
    dataset = ImageFolderTrackDataset(test_dir,
                                      transform=transform)
    sampler = TrackSampler(dataset,
                           samples_per_class)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             num_workers=8,
                                             batch_sampler=sampler)

    return dataloader