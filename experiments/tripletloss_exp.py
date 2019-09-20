
import utils
from dataset_utils import coxs2v, vggface2, bbt, mnist, usps, mnistBig, uspsBig
from dataset_utils import dataloaders
from torchvision import transforms

import torch


def Get_TrainDataloaders(exp_name, config):

    transfrom_list = []
    transfrom_list.append(
        transforms.Resize((config.hyperparameters.image_size, config.hyperparameters.image_size), interpolation=1))
    if config.hyperparameters.random_hor_flip:
        transfrom_list.append(transforms.RandomHorizontalFlip())
    transfrom_list.append(transforms.ToTensor())
    train_transforms = transforms.Compose(transfrom_list)

    eval_transforms = transforms.Compose([
        transforms.Resize((config.hyperparameters.image_size, config.hyperparameters.image_size), interpolation=1),
        transforms.ToTensor()
    ])

    test_batch_size = (config.hyperparameters.people_per_batch * config.hyperparameters.images_per_person) // 2
    nrof_folds = config.dataset.cross_validation.num_fold
    fold_tool = utils.FoldGenerator(nrof_folds,
                                    config.dataset.cross_validation.num_train_folds,
                                    config.dataset.cross_validation.num_val_folds)
    train_folds, val_folds, test_folds = fold_tool.get_fold()

    if exp_name == 'train_vggface2':
        train_loader = vggface2.get_vggface2_trainset(config.dataset.vggface2.train_dir,
                                                      train_transforms,
                                                      config.hyperparameters.people_per_batch,
                                                      config.hyperparameters.images_per_person)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         ['lfw'])

    elif exp_name == 'COXvideo1_finetune':
        train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video1_dir,
                                                  config.dataset.coxs2v.video1_pairs,
                                                  train_folds,
                                                  nrof_folds,
                                                  train_transforms,
                                                  config.hyperparameters.people_per_batch,
                                                  config.hyperparameters.images_per_person)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         ['lfw', 'cox_video1'])

    elif exp_name == 'COXvideo1_calibration':
        train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video1_dir,
                                                  config.dataset.coxs2v.video1_pairs,
                                                  val_folds,
                                                  nrof_folds,
                                                  train_transforms,
                                                  config.hyperparameters.people_per_batch,
                                                  config.hyperparameters.images_per_person,
                                                  video_only=True)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         ['lfw', 'cox_video1'])

    elif exp_name == 'COXvideo2_finetune':
        train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video2_dir,
                                                  config.dataset.coxs2v.video2_pairs,
                                                  train_folds,
                                                  nrof_folds,
                                                  train_transforms,
                                                  config.hyperparameters.people_per_batch,
                                                  config.hyperparameters.images_per_person)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         ['lfw', 'cox_video2'])

    elif exp_name == 'COXvideo2_calibration':
        train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video2_dir,
                                                  config.dataset.coxs2v.video2_pairs,
                                                  val_folds,
                                                  nrof_folds,
                                                  train_transforms,
                                                  config.hyperparameters.people_per_batch,
                                                  config.hyperparameters.images_per_person,
                                                  video_only=True)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         ['lfw', 'cox_video2'])

    elif exp_name == 'COXvideo3_finetune':
        train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video3_dir,
                                                  config.dataset.coxs2v.video3_pairs,
                                                  train_folds,
                                                  nrof_folds,
                                                  train_transforms,
                                                  config.hyperparameters.people_per_batch,
                                                  config.hyperparameters.images_per_person)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         ['lfw', 'cox_video3'])

    elif exp_name == 'COXvideo3_calibration':
        train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video3_dir,
                                                  config.dataset.coxs2v.video3_pairs,
                                                  val_folds,
                                                  nrof_folds,
                                                  train_transforms,
                                                  config.hyperparameters.people_per_batch,
                                                  config.hyperparameters.images_per_person,
                                                  video_only=True)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         ['lfw', 'cox_video3'])

    elif exp_name == 'COXvideo4_finetune':
        train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video4_dir,
                                                  config.dataset.coxs2v.video4_pairs,
                                                  train_folds,
                                                  nrof_folds,
                                                  train_transforms,
                                                  config.hyperparameters.people_per_batch,
                                                  config.hyperparameters.images_per_person)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         ['lfw', 'cox_video4'])

    elif exp_name == 'COXvideo4_calibration':
        train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video4_dir,
                                                  config.dataset.coxs2v.video4_pairs,
                                                  val_folds,
                                                  nrof_folds,
                                                  train_transforms,
                                                  config.hyperparameters.people_per_batch,
                                                  config.hyperparameters.images_per_person,
                                                  video_only=True)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         ['lfw', 'cox_video4'])

    elif exp_name == 'BBT_trackfinetune_GT':

        train_set = bbt.BBTTrackDataset(config.dataset.bbt.annotation_path,
                                        config.dataset.bbt.movie_path,
                                        100000,
                                        transform=train_transforms)

        batch_sampler = bbt.TrackSampler(train_set,
                                         config.hyperparameters.images_per_person)

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   num_workers=8,
                                                   batch_sampler=batch_sampler,
                                                   pin_memory=True)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         ['lfw'])

    elif exp_name == 'BBT_trackfinetune':

        train_loader = bbt.get_trainset(config.dataset.bbt.pkl_dataset_path,
                                        config.hyperparameters.images_per_person,
                                        transform=train_transforms)

        test_container = dataloaders.get_testdataloaders(config,
                                                         eval_transforms,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         [])

    elif exp_name == 'mnist':

        train_loader = mnist.get_mnist("train",
                                       batch_size=config.hyperparameters.batch_size)
        test_container = [('mnist',
                          mnist.get_mnist("val",
                                          batch_size=config.hyperparameters.batch_size))]

    else:
        raise Exception('Experiment {} does not exist.'.format(exp_name))

    return train_loader, test_container
