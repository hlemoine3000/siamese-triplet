
import os

import torch

import utils
import evaluation
from dataset_utils import dataset
from dataset_utils import sampler


video_options = ['cox_video1', 'cox_video2', 'cox_video3', 'cox_video4']


def get_coxs2v_trainset(video_chosen: str,
                        cox_paths_dict: dict,
                        folds,
                        nrof_folds,
                        data_transform,
                        people_per_batch,
                        images_per_person,
                        num_workers=4,
                        balanced=True,
                        video_only=False,
                        samples_division_list=None,  # [0.6, 0.4]
                        div_idx: int = -1):

    if not video_chosen in video_options:
        raise Exception('COX video option {} does not exist. Must be one option of the following list {}.'.format(video_chosen, video_options))

    still_dir = cox_paths_dict['still_dir']
    video_dir = cox_paths_dict['{}_dir'.format(video_chosen)]
    pairs_file = cox_paths_dict['{}_pairs'.format(video_chosen)]

    # Set up train loader
    print('TRAIN SET COXS2V:\t{}'.format(video_dir))
    train_set = dataset.DatasetS2V(still_dir,
                                   video_dir,
                                   pairs_file,
                                   data_transform,
                                   folds,
                                   preload=True,
                                   video_only=video_only,
                                   num_folds=nrof_folds,
                                   samples_division_list=samples_division_list,  # [0.4, 0.6]
                                   div_idx=div_idx)

    if balanced:
        batch_sampler = sampler.Random_S2VBalancedBatchSampler(train_set,
                                                       people_per_batch,
                                                       images_per_person,
                                                       max_batches=1000)

        return torch.utils.data.DataLoader(train_set,
                                           num_workers=num_workers,
                                           batch_sampler=batch_sampler)
    else:
        return torch.utils.data.DataLoader(train_set,
                                           batch_size=people_per_batch*images_per_person,
                                           shuffle=True,
                                           num_workers=num_workers)

def get_evaluator(video_chosen: str,
                  cox_paths_dict: dict,
                  folds: list,
                  nrof_folds: int,
                  data_transform,
                  batch_size: int,
                  num_workers: int=4,
                  preload: bool=False) -> evaluation.Evaluator:

    if not video_chosen in video_options:
        raise Exception('COX video option {} does not exist. Must be one option of the following list {}.'.format(video_chosen, video_options))

    still_dir = cox_paths_dict['still_dir']
    video_dir = cox_paths_dict['{}_dir'.format(video_chosen)]
    pairs_file = cox_paths_dict['{}_pairs'.format(video_chosen)]

    print('TEST SET COXS2V:\t{}'.format(video_dir))
    test_set = dataset.PairsDatasetS2V(still_dir,
                                       video_dir,
                                       pairs_file,
                                       data_transform,
                                       folds,
                                       num_folds=nrof_folds,
                                       preload=preload)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=num_workers, batch_size=batch_size)
    return evaluation.Pairs_Evaluator(test_loader, video_chosen)


def get_coxs2v_samples(still_dir, video_dir, video_list, subject_list, max_samples_per_subject=10, video_only=False):

    image_path_list = []
    labels_list = []

    for video_view in video_list:
        for subject in subject_list:

            subject_video_path = os.path.join(video_dir, video_view, subject)
            video_image_paths = utils.get_image_paths(subject_video_path)
            if len(video_image_paths) > max_samples_per_subject:
                video_image_paths = video_image_paths[0:max_samples_per_subject]

            label = subject + '_' + video_view
            labels_list += [label] * len(video_image_paths)

            if not video_only:
                video_image_paths += os.path.join(still_dir, subject + '_0000.JPG')
                labels_list += [subject + '_still']

            image_path_list += video_image_paths

    label_name_list = utils.unique(labels_list)
    label_to_target_dict = {key: tgt for (tgt, key) in enumerate(label_name_list)}

    target_list = []
    for label in labels_list:
        target_list += label_to_target_dict[label]

    return image_path_list, target_list, label_name_list
