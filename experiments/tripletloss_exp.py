
import utils
from dataset_utils import coxs2v, vggface2
from dataset_utils import dataloaders


def Get_TrainDataloaders(exp_name, config, data_transform):

    test_batch_size = (config.hyperparameters.people_per_batch * config.hyperparameters.images_per_person) // 2
    nrof_folds = config.dataset.cross_validation.num_fold
    fold_tool = utils.FoldGenerator(nrof_folds,
                                    config.dataset.cross_validation.num_train_folds,
                                    config.dataset.cross_validation.num_val_folds)
    train_folds, val_folds, test_folds = fold_tool.get_fold()

    if exp_name == 'train_vggface2':
        train_loader = vggface2.get_vggface2_trainset(config.dataset.vggface2.train_dir,
                                                      data_transform,
                                                      config.hyperparameters.people_per_batch,
                                                      config.hyperparameters.images_per_person)

        test_container = dataloaders.Get_TestDataloaders(config,
                                                         data_transform,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         is_vggface2=False,
                                                         is_lfw=True,
                                                         is_cox_video1=False,
                                                         is_cox_video2=False,
                                                         is_cox_video3=False,
                                                         is_cox_video4=False)

    elif exp_name == 'COXvideo2_finetune':
        train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video2_dir,
                                                  config.dataset.coxs2v.video2_pairs,
                                                  train_folds,
                                                  nrof_folds,
                                                  data_transform,
                                                  config.hyperparameters.people_per_batch,
                                                  config.hyperparameters.images_per_person)

        test_container = dataloaders.Get_TestDataloaders(config,
                                                         data_transform,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         is_vggface2=False,
                                                         is_lfw=True,
                                                         is_cox_video1=True,
                                                         is_cox_video2=True,
                                                         is_cox_video3=True,
                                                         is_cox_video4=True)

    elif exp_name == 'COXvideo3_finetune':
        train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                  config.dataset.coxs2v.video3_dir,
                                                  config.dataset.coxs2v.video3_pairs,
                                                  train_folds,
                                                  nrof_folds,
                                                  data_transform,
                                                  config.hyperparameters.people_per_batch,
                                                  config.hyperparameters.images_per_person)

        test_container = dataloaders.Get_TestDataloaders(config,
                                                         data_transform,
                                                         test_batch_size,
                                                         test_folds,
                                                         nrof_folds,
                                                         is_vggface2=False,
                                                         is_lfw=True,
                                                         is_cox_video1=True,
                                                         is_cox_video2=True,
                                                         is_cox_video3=True,
                                                         is_cox_video4=True)

    else:
        raise Exception('Experiment {} does not exist.'.format(exp_name))

    return train_loader, test_container