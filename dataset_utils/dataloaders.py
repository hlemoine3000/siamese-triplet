
import torch
from torch.utils import data

import dataset_utils
from dataset_utils import sampler, dataset
from dataset_utils import bbt, coxs2v, lfw, vggface2, mnistBig, mnist, usps, uspsBig
import evaluation

def Get_ImageFolderLoader(data_dir,
                 data_transform,
                 people_per_batch,
                 images_per_person):


    train_set = dataset.DatasetFolder(data_dir, transform=data_transform)

    batch_sampler = sampler.Random_BalancedBatchSampler(train_set,
                                                        people_per_batch,
                                                        images_per_person,
                                                        max_batches=1000)

    return torch.utils.data.DataLoader(train_set,
                                       num_workers=4,
                                       batch_sampler=batch_sampler,
                                       pin_memory=True)


def Get_PairsImageFolderLoader(data_dir,
                               pairs_file,
                               data_transform,
                               batch_size,
                               preload=False):

    num_workers = 2 if preload else 8

    test_set = dataset_utils.dataset.PairsDataset(data_dir,
                                     pairs_file,
                                     transform=data_transform,
                                     preload=preload)
    return torch.utils.data.DataLoader(test_set, num_workers=num_workers, batch_size=batch_size)


def get_testdataloaders(config,
                        data_transform,
                        batch_size,
                        folds,
                        nrof_folds,
                        dataset_list):

    test_loaders_list = []

    # VGGFACE2 dataset
    if 'vggface2' in dataset_list:
        data_loader = vggface2.get_vggface2_testset(config.dataset.vggface2.test_dir,
                                           config.dataset.vggface2.pairs_file,
                                           data_transform,
                                           batch_size,
                                           preload=False)
        test_loaders_list.append(('vggface2', data_loader, evaluation.pair_evaluate))

    # LFW dataset
    if 'lfw' in dataset_list:
        data_loader = lfw.get_lfw_testset(config.dataset.lfw.test_dir,
                                      config.dataset.lfw.pairs_file,
                                      data_transform,
                                      batch_size,
                                      preload=False)
        test_loaders_list.append(('lfw', data_loader, evaluation.pair_evaluate))

    # COXS2V dataset
    if 'cox_video1' in dataset_list:
        data_loader = coxs2v.get_coxs2v_testset(config.dataset.coxs2v.still_dir,
                                                config.dataset.coxs2v.video1_dir,
                                                config.dataset.coxs2v.video1_pairs,
                                                folds,
                                                nrof_folds,
                                                data_transform,
                                                batch_size,
                                                preload=True)
        test_loaders_list.append(('cox_video1', data_loader, evaluation.pair_evaluate))

    if 'cox_video2' in dataset_list:
        data_loader = coxs2v.get_coxs2v_testset(config.dataset.coxs2v.still_dir,
                                                config.dataset.coxs2v.video2_dir,
                                                config.dataset.coxs2v.video2_pairs,
                                                folds,
                                                nrof_folds,
                                                data_transform,
                                                batch_size,
                                                preload=True)
        test_loaders_list.append(('cox_video2', data_loader, evaluation.pair_evaluate))

    if 'cox_video3' in dataset_list:
        data_loader = coxs2v.get_coxs2v_testset(config.dataset.coxs2v.still_dir,
                                                config.dataset.coxs2v.video3_dir,
                                                config.dataset.coxs2v.video3_pairs,
                                                folds,
                                                nrof_folds,
                                                data_transform,
                                                batch_size,
                                                preload=True)
        test_loaders_list.append(('cox_video3', data_loader, evaluation.pair_evaluate))

    if 'cox_video4' in dataset_list:
        data_loader = coxs2v.get_coxs2v_testset(config.dataset.coxs2v.still_dir,
                                                config.dataset.coxs2v.video4_dir,
                                                config.dataset.coxs2v.video4_pairs,
                                                folds,
                                                nrof_folds,
                                                data_transform,
                                                batch_size,
                                                preload=True)
        test_loaders_list.append(('cox_video4', data_loader, evaluation.pair_evaluate))

    if 'bbt_ep01' in dataset_list:
        data_loader = bbt.get_testset(config.dataset.bbt.video_track_path,
                                      data_transform,
                                      batch_size)
        test_loaders_list.append(('bbt_ep01', data_loader, evaluation.video_description_evaluate))

    if 'mnist' in dataset_list:
        data_loader = mnist.get_mnist("val",
                                      batch_size=config.hyperparameters.batch_size)
        test_loaders_list.append(('cox_video4', data_loader, evaluation.pair_evaluate))

    return test_loaders_list


def get_traindataloaders(config,
                         data_transform,
                         people_per_batch,
                         images_per_person,
                         folds,
                         nrof_folds,
                         is_vggface2=False,
                         is_cox_video1=False,
                         is_cox_video2=False,
                         is_cox_video3=False,
                         is_cox_video4=False):

    train_loaders_list = []

    # VGGFACE2 dataset
    if is_vggface2:
        data_loader = vggface2.get_vggface2_trainset(config.dataset.vggface2.train_dir,
                                                     data_transform,
                                                     people_per_batch,
                                                     images_per_person)
        train_loaders_list.append(('vggface2', data_loader))

    # COXS2V dataset
    if is_cox_video1:
        data_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                 config.dataset.coxs2v.video1_dir,
                                                 config.dataset.coxs2v.video1_pairs,
                                                 folds,
                                                 nrof_folds,
                                                 data_transform,
                                                 people_per_batch,
                                                 images_per_person)
        train_loaders_list.append(('cox_video1', data_loader))

    if is_cox_video2:
        data_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                 config.dataset.coxs2v.video2_dir,
                                                 config.dataset.coxs2v.video2_pairs,
                                                 folds,
                                                 nrof_folds,
                                                 data_transform,
                                                 people_per_batch,
                                                 images_per_person)
        train_loaders_list.append(('cox_video2', data_loader))

    if is_cox_video3:
        data_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                 config.dataset.coxs2v.video3_dir,
                                                 config.dataset.coxs2v.video3_pairs,
                                                 folds,
                                                 nrof_folds,
                                                 data_transform,
                                                 people_per_batch,
                                                 images_per_person)
        train_loaders_list.append(('cox_video3', data_loader))

    if is_cox_video4:
        data_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                 config.dataset.coxs2v.video4_dir,
                                                 config.dataset.coxs2v.video4_pairs,
                                                 folds,
                                                 nrof_folds,
                                                 data_transform,
                                                 people_per_batch,
                                                 images_per_person)
        train_loaders_list.append(('cox_video4', data_loader))

    return train_loaders_list