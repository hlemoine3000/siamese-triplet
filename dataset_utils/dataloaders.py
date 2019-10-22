
import torch
from torch.utils import data
from torchvision import transforms

import dataset_utils
import utils


def get_image_folder_loader(data_dir,
                            data_transform,
                            people_per_batch,
                            images_per_person,
                            num_workers=8):

    train_set = dataset_utils.dataset.DatasetFolder(data_dir, transform=data_transform)

    batch_sampler = dataset_utils.sampler.Random_BalancedBatchSampler(train_set,
                                                        people_per_batch,
                                                        images_per_person,
                                                        max_batches=1000)

    return torch.utils.data.DataLoader(train_set,
                                       num_workers=num_workers,
                                       batch_sampler=batch_sampler)


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


def get_evaluators(dataset_list: list,
                       config) -> list:

    evaluators_list = []

    eval_transforms = transforms.Compose([
        transforms.Resize((config.hyperparameters.image_size, config.hyperparameters.image_size), interpolation=1),
        transforms.ToTensor()
    ])

    fold_tool = utils.FoldGenerator(config.dataset.cross_validation.num_fold,
                                    config.dataset.cross_validation.num_train_folds,
                                    config.dataset.cross_validation.num_val_folds)
    train_folds, val_folds, test_folds = fold_tool.get_fold()

    for dataset_name in dataset_list:
        # VGGFACE2 dataset
        if 'vggface2' == dataset_name:
            evaluator = dataset_utils.vggface2.get_vggface2_evaluator(config.dataset.vggface2.test_dir,
                                                          config.dataset.vggface2.pairs_file,
                                                          eval_transforms,
                                                          config.hyperparameters.batch_size,
                                                          preload=False)
        # LFW dataset
        elif 'lfw' == dataset_name:
            evaluator = dataset_utils.lfw.get_evaluator(config.dataset.lfw,
                                          eval_transforms,
                                          config.hyperparameters.batch_size,
                                          preload=False)
        # COXS2V dataset
        elif 'cox_video' in dataset_name:
            evaluator = dataset_utils.coxs2v.get_evaluator(dataset_name,
                                             config.dataset.coxs2v,
                                             test_folds,
                                             config.dataset.cross_validation.num_fold,
                                             eval_transforms,
                                             config.hyperparameters.batch_size,
                                             preload=True)
        # BBT generated dataset
        elif 'bbt' == dataset_name:
            from dataset_utils import bbt
            evaluator = bbt.get_evaluator(config.dataset.bbt.dataset_path,
                                            eval_transforms,
                                            config.hyperparameters.batch_size)
        # MNIST dataset
        elif 'mnist' == dataset_name:
            evaluator = dataset_utils.mnist.get_evaluator(batch_size=config.hyperparameters.batch_size)
        # USPS dataset
        elif 'usps' == dataset_name:
            evaluator = dataset_utils.usps.get_evaluator(batch_size=config.hyperparameters.batch_size)
        else:
            raise Exception('Datatset {} not supported.'.format(dataset_name))

        evaluators_list.append(evaluator)

    return evaluators_list


def get_traindataloaders(dataset_name: str,
                         config) -> data.DataLoader:

    transfrom_list = [transforms.Resize((config.hyperparameters.image_size, config.hyperparameters.image_size), interpolation=1)]
    if config.hyperparameters.random_hor_flip:
        transfrom_list.append(transforms.RandomHorizontalFlip())
    transfrom_list.append(transforms.ToTensor())
    train_transforms = transforms.Compose(transfrom_list)

    fold_tool = utils.FoldGenerator(config.dataset.cross_validation.num_fold,
                                    config.dataset.cross_validation.num_train_folds,
                                    config.dataset.cross_validation.num_val_folds)
    train_folds, val_folds, test_folds = fold_tool.get_fold()

    # VGGFACE2 dataset
    if dataset_name == 'vggface2':
        data_loader = dataset_utils.vggface2.get_vggface2_trainset(config.dataset.vggface2.train_dir,
                                                     train_transforms,
                                                     config.hyperparameters.people_per_batch,
                                                     config.hyperparameters.images_per_person)
    # COXS2V dataset
    elif 'cox_video' in dataset_name:
        data_loader = dataset_utils.coxs2v.get_coxs2v_trainset(dataset_name,
                                                 config.dataset.coxs2v,
                                                 train_folds,
                                                 config.dataset.cross_validation.num_fold,
                                                 train_transforms,
                                                 config.hyperparameters.people_per_batch,
                                                 config.hyperparameters.images_per_person)
    # BBT generated dataset
    elif dataset_name == 'bbt':
        data_loader = dataset_utils.bbt.get_trainset(config.dataset.bbt.dataset_path,
                                       config.hyperparameters.images_per_person,
                                       transform=train_transforms)
    # Movie generated dataset
    elif dataset_name == 'movie':
        data_loader = dataset_utils.movie.get_trainset(config.dataset.movie.dataset_path,
                                                     config.hyperparameters.images_per_person,
                                                     transform=train_transforms)
    # MNIST dataset
    elif dataset_name == "mnist":
        data_loader = dataset_utils.mnist.get_mnist_trainset(batch_size=config.hyperparameters.batch_size,
                                               small=True)
    elif dataset_name == "mnistbig":
        data_loader = dataset_utils.mnist.get_mnist_trainset(batch_size=config.hyperparameters.batch_size,
                                               small=False)
    # USPS dataset
    elif dataset_name == 'usps':
        data_loader = dataset_utils.usps.get_usps_trainset(batch_size=config.hyperparameters.batch_size,
                                             small=True)
    elif dataset_name == 'uspsbig':
        data_loader = dataset_utils.usps.get_usps_trainset(batch_size=config.hyperparameters.batch_size,
                                             small=False)
    else:
        raise Exception('Datatset {} not supported.'.format(dataset_name))

    return data_loader