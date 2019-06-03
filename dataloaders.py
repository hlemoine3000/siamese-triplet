
import torch

import utils
from dataset_utils import PairsDataset, PairsDatasetS2V, DatasetS2V
from dataset_utils import Random_BalancedBatchSampler, Random_S2VBalancedBatchSampler
from torchvision import transforms, datasets

def data_loaders(config, data_transform):

    if (config.dataset.dataloader == "base"):
        return base_loader(config, data_transform)
    elif (config.dataset.dataloader == "coxfinetune"):
        return coxfinetune_loader(config, data_transform)
    else:
        raise Exception("Dataloader of name {} does not exist.".format(config.dataset.dataloader))

def base_loader(config, data_transform):

    test_loader_list = []
    test_batch_size = (config.hyperparameters.people_per_batch * config.hyperparameters.images_per_person) // 2
    nrof_folds = config.dataset.cross_validation.num_fold

    # Set up test loader
    for i, test_name in enumerate(config.dataset.test_name):
        # Test loader
        print('TEST SET {}:\t{}'.format(test_name, config.dataset.test_dir[i]))
        test_set = PairsDataset(config.dataset.test_dir[i], config.dataset.pairs_file[i], transform=data_transform, preload=True)
        test_loader = torch.utils.data.DataLoader(test_set, num_workers=2, batch_size=test_batch_size)
        test_loader_list.append((test_name, test_loader, int(nrof_folds)))

    if config.dataset.is_s2v_dataset:
        fold_tool = utils.FoldGenerator(nrof_folds,
                                        config.dataset.cross_validation.num_train_folds,
                                        config.dataset.cross_validation.num_val_folds)
        train_folds, val_folds, test_folds = fold_tool.get_fold()
        for i, set_name in enumerate(config.dataset.s2v_dataset.names):
            # Test loader
            print('TEST SET {}:\t{}'.format(set_name, config.dataset.s2v_dataset.video_dir[i]))
            test_set = PairsDatasetS2V(config.dataset.s2v_dataset.still_dir,
                                       config.dataset.s2v_dataset.video_dir[i],
                                       config.dataset.s2v_dataset.pairs_file[i],
                                       data_transform,
                                       test_folds,
                                       num_folds=nrof_folds,
                                       preload=True)
            test_loader = torch.utils.data.DataLoader(test_set, num_workers=2, batch_size=test_batch_size)
            test_loader_list.append((set_name, test_loader, int(nrof_folds)))

    # Set up train loader
    print('TRAIN SET {}:\t{}'.format(config.dataset.train_name, config.dataset.train_dir))
    train_set = datasets.ImageFolder(config.dataset.train_dir, transform=data_transform)

    batch_sampler = Random_BalancedBatchSampler(train_set, config.hyperparameters.people_per_batch,
                                                config.hyperparameters.images_per_person, max_batches=1000)
    train_loader = torch.utils.data.DataLoader(train_set,
                                                      num_workers=8,
                                                      batch_sampler=batch_sampler,
                                                      pin_memory=True)

    return train_loader, test_loader_list

def coxfinetune_loader(config, data_transform):



    test_loader_list = []
    test_batch_size = (config.hyperparameters.people_per_batch * config.hyperparameters.images_per_person) // 2
    nrof_folds = config.dataset.cross_validation.num_fold

    fold_tool = utils.FoldGenerator(nrof_folds,
                                    config.dataset.cross_validation.num_train_folds,
                                    config.dataset.cross_validation.num_val_folds)
    train_folds, val_folds, test_folds = fold_tool.get_fold()

    # Set up train loader
    print('TRAIN SET {}'.format('COX-S2V'))

    train_set = DatasetS2V(config.dataset.coxs2v.still_dir,
                           config.dataset.coxs2v.video2_dir,
                           config.dataset.coxs2v.video2_pairs,
                           data_transform,
                           train_folds,
                           num_folds=nrof_folds)

    batch_sampler = Random_S2VBalancedBatchSampler(train_set, config.hyperparameters.people_per_batch,
                                                config.hyperparameters.images_per_person, max_batches=1000)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               num_workers=8,
                                               batch_sampler=batch_sampler,
                                               pin_memory=True)


    # Set up test loader
    for i, test_name in enumerate(config.dataset.test_name):
        # Test loader
        print('TEST SET {}:\t{}'.format(test_name, config.dataset.test_dir[i]))
        test_set = PairsDataset(config.dataset.test_dir[i], config.dataset.pairs_file[i], transform=data_transform, preload=True)
        test_loader = torch.utils.data.DataLoader(test_set, num_workers=2, batch_size=test_batch_size)
        test_loader_list.append((test_name, test_loader, int(nrof_folds)))

    if config.dataset.is_s2v_dataset:



        for i, set_name in enumerate(config.dataset.s2v_dataset.names):
            # Test loader
            print('TEST SET {}:\t{}'.format(set_name, config.dataset.s2v_dataset.video_dir[i]))
            test_set = PairsDatasetS2V(config.dataset.s2v_dataset.still_dir,
                                       config.dataset.s2v_dataset.video_dir[i],
                                       config.dataset.s2v_dataset.pairs_file[i],
                                       data_transform,
                                       test_folds,
                                       num_folds=nrof_folds,
                                       preload=True)
            test_loader = torch.utils.data.DataLoader(test_set, num_workers=2, batch_size=test_batch_size)
            test_loader_list.append((set_name, test_loader, int(nrof_folds)))




    return train_loader, test_loader_list

def domain_adaptation_loader(config, data_transform):

    test_loader_list = []
    target_batch_size = config.hyperparameters.people_per_batch * config.hyperparameters.images_per_person
    test_batch_size = (config.hyperparameters.people_per_batch * config.hyperparameters.images_per_person) // 2
    nrof_folds = config.dataset.cross_validation.num_fold

    fold_tool = utils.FoldGenerator(nrof_folds,
                                    config.dataset.cross_validation.num_train_folds,
                                    config.dataset.cross_validation.num_val_folds)
    train_folds, val_folds, test_folds = fold_tool.get_fold()

    # Set up train loader
    print('TRAIN SET {}'.format('COX-S2V'))

    ##########
    # Source #
    ##########
    source_train_set = DatasetS2V(config.dataset.coxs2v.still_dir,
                                  config.dataset.coxs2v.video2_dir,
                                  config.dataset.coxs2v.video2_pairs,
                                  data_transform,
                                  train_folds,
                                  num_folds=nrof_folds,
                                  video_only=False)

    batch_sampler = Random_BalancedBatchSampler(source_train_set, config.hyperparameters.people_per_batch,
                                                   config.hyperparameters.images_per_person, max_batches=1000)
    source_train_loader = torch.utils.data.DataLoader(source_train_set,
                                               num_workers=8,
                                               batch_sampler=batch_sampler,
                                               pin_memory=True)

    ##########
    # Target #
    ##########
    target_train_set = DatasetS2V(config.dataset.coxs2v.still_dir,
                                  config.dataset.coxs2v.video2_dir,
                                  config.dataset.coxs2v.video2_pairs,
                                  data_transform,
                                  val_folds,
                                  num_folds=nrof_folds,
                                  video_only=True)

    target_train_loader = torch.utils.data.DataLoader(target_train_set,
                                                      num_workers=8,
                                                      batch_size=target_batch_size,
                                                      shuffle=True,
                                                      pin_memory=True)

    ##########
    # Test #
    ##########

    # Test loader
    test_set = PairsDatasetS2V(config.dataset.coxs2v.still_dir,
                               config.dataset.coxs2v.video2_dir,
                               config.dataset.coxs2v.video2_pairs,
                               data_transform,
                               test_folds,
                               num_folds=nrof_folds,
                               preload=True)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=2, batch_size=test_batch_size)
    test_loader_list.append(('video2_test', test_loader, int(nrof_folds)))

    return source_train_loader, target_train_loader, test_loader_list