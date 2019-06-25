
import utils
from dataset_utils import coxs2v, vggface2
from dataset_utils import dataloaders
from torchvision import transforms


def Get_DADataloaders(exp_name, config):

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

    if exp_name == 'da_video2_to_video4':
        source_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                          config.dataset.coxs2v.video2_dir,
                                                          config.dataset.coxs2v.video2_pairs,
                                                          train_folds,
                                                          nrof_folds,
                                                          train_transforms,
                                                          config.hyperparameters.people_per_batch,
                                                          config.hyperparameters.images_per_person)

        target_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                   config.dataset.coxs2v.video4_dir,
                                                   config.dataset.coxs2v.video4_pairs,
                                                   val_folds,
                                                   nrof_folds,
                                                   train_transforms,
                                                   config.hyperparameters.people_per_batch,
                                                   config.hyperparameters.images_per_person,
                                                   video_only=True)

        test_loaders_list = dataloaders.Get_TestDataloaders(config,
                                                            eval_transforms,
                                                            test_batch_size,
                                                            test_folds,
                                                            nrof_folds,
                                                            is_vggface2=False,
                                                            is_lfw=True,
                                                            is_cox_video1=False,
                                                            is_cox_video2=True,
                                                            is_cox_video3=False,
                                                            is_cox_video4=True)

    elif exp_name == 'da_video2_to_video3':
        source_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                          config.dataset.coxs2v.video2_dir,
                                                          config.dataset.coxs2v.video2_pairs,
                                                          train_folds,
                                                          nrof_folds,
                                                          train_transforms,
                                                          config.hyperparameters.people_per_batch,
                                                          config.hyperparameters.images_per_person)

        target_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                   config.dataset.coxs2v.video3_dir,
                                                   config.dataset.coxs2v.video3_pairs,
                                                   val_folds,
                                                   nrof_folds,
                                                   train_transforms,
                                                   config.hyperparameters.people_per_batch,
                                                   config.hyperparameters.images_per_person,
                                                   video_only=True)

        test_loaders_list = dataloaders.Get_TestDataloaders(config,
                                                            eval_transforms,
                                                            test_batch_size,
                                                            test_folds,
                                                            nrof_folds,
                                                            is_vggface2=False,
                                                            is_lfw=True,
                                                            is_cox_video1=False,
                                                            is_cox_video2=True,
                                                            is_cox_video3=True,
                                                            is_cox_video4=False)

    elif exp_name == 'da_vggface2_to_video2':
        source_loader = vggface2.get_vggface2_trainset(config.dataset.vggface2.train_dir,
                                                       train_transforms,
                                                       config.hyperparameters.people_per_batch,
                                                       config.hyperparameters.images_per_person)

        target_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                   config.dataset.coxs2v.video2_dir,
                                                   config.dataset.coxs2v.video2_pairs,
                                                   train_folds,
                                                   nrof_folds,
                                                   train_transforms,
                                                   config.hyperparameters.people_per_batch,
                                                   config.hyperparameters.images_per_person,
                                                   video_only=True)

        test_loaders_list = dataloaders.Get_TestDataloaders(config,
                                                            eval_transforms,
                                                            test_batch_size,
                                                            test_folds,
                                                            nrof_folds,
                                                            is_vggface2=False,
                                                            is_lfw=True,
                                                            is_cox_video1=False,
                                                            is_cox_video2=True,
                                                            is_cox_video3=False,
                                                            is_cox_video4=False)

    elif exp_name == 'da_vggface2_to_video3':
        source_loader = vggface2.get_vggface2_trainset(config.dataset.vggface2.train_dir,
                                                       train_transforms,
                                                       config.hyperparameters.people_per_batch,
                                                       config.hyperparameters.images_per_person)

        target_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                   config.dataset.coxs2v.video3_dir,
                                                   config.dataset.coxs2v.video3_pairs,
                                                   train_folds,
                                                   nrof_folds,
                                                   train_transforms,
                                                   config.hyperparameters.people_per_batch,
                                                   config.hyperparameters.images_per_person,
                                                   video_only=True)

        test_loaders_list = dataloaders.Get_TestDataloaders(config,
                                                            eval_transforms,
                                                            test_batch_size,
                                                            test_folds,
                                                            nrof_folds,
                                                            is_vggface2=False,
                                                            is_lfw=True,
                                                            is_cox_video1=False,
                                                            is_cox_video2=False,
                                                            is_cox_video3=True,
                                                            is_cox_video4=False)

    else:
        raise Exception('Experiment {} does not exist.'.format(exp_name))

    return source_loader, target_loader, test_loaders_list