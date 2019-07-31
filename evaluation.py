
import argparse
import sys
import numpy as np
import tqdm
import pandas as pd
import json
from sklearn import metrics

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from copy import deepcopy
import utils
import models
from dataset_utils.dataloaders import Get_TestDataloaders


def Evaluate(test_loader: DataLoader,
             model,
             device,
             step,
             plotter=None,
             name='validation',
             nrof_folds=10,
             distance_metric=0,
             val_far=1e-3,
             plot_distances=False):
    embeddings1 = []
    embeddings2 = []
    issame_array = []

    model.eval()

    with torch.no_grad():
        tbar = tqdm.tqdm(test_loader, dynamic_ncols=True)
        for images_batch, issame in tbar:
            # Transfer to GPU

            image_batch1 = images_batch[0].to(device, non_blocking=True)
            image_batch2 = images_batch[1].to(device, non_blocking=True)

            emb1 = model.forward(image_batch1)
            emb2 = model.forward(image_batch2)

            embeddings1.append(emb1)
            embeddings2.append(emb2)
            issame_array.append(deepcopy(issame))

        embeddings1 = torch.cat(embeddings1, 0).cpu().numpy()
        embeddings2 = torch.cat(embeddings2, 0).cpu().numpy()
        issame_array = torch.cat(issame_array, 0).cpu().numpy()

    distance_and_is_same = zip(np.sum((embeddings1 - embeddings2) ** 2, axis=1), issame_array)
    distance_and_is_same_df = pd.DataFrame(distance_and_is_same)
    negative_mean_distance = distance_and_is_same_df[distance_and_is_same_df[1] == False][0].mean()
    positive_mean_distance = distance_and_is_same_df[distance_and_is_same_df[1] == True][0].mean()

    thresholds = np.arange(0, 4, 0.01)
    subtract_mean = False

    tpr, fpr, accuracy = utils.Calculate_Roc(thresholds, embeddings1, embeddings2,
                                                             np.asarray(issame_array), nrof_folds=nrof_folds,
                                                             distance_metric=distance_metric,
                                                             subtract_mean=subtract_mean)

    # thresholds = np.arange(0, 4, 0.001)
    # val, val_std, far, threshold_lowfar = utils.Calculate_Val(thresholds, embeddings1, embeddings2,
    #                                                           np.asarray(issame_array),
    #                                                           val_far,
    #                                                           nrof_folds=nrof_folds,
    #                                                           distance_metric=distance_metric,
    #                                                           subtract_mean=subtract_mean)

    print('Accuracy: {:.3%}+-{:.3%}'.format(np.mean(accuracy), np.std(accuracy)))
    # print('Validation rate: {:.3%}+-{:.3%} @ FAR={:.3%}'.format(val, val_std, far))
    print('Positive mean distances: {:.3}'.format(positive_mean_distance))
    print('Negative mean distances: {:.3}'.format(negative_mean_distance))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.5f' % auc)

    # if plotter:
    #     if plot_distances:
    #         plotter.plot('distance', 'step', name + '_an', 'Pairwise mean distance', step, negative_mean_distance)
    #         plotter.plot('distance', 'step', name + '_ap', 'Pairwise mean distance', step, positive_mean_distance)
    #
    #     plotter.plot('accuracy', 'step', name, 'Accuracy', step, np.mean(accuracy))
    #     # plotter.plot('validation rate', 'step', name, 'Validation Rate @ FAR={:.3%}'.format(val_far), step, val)
    #     plotter.plot('auc', 'step', name, 'AUC', step, auc)

    data_dict = {'accuracy': np.mean(accuracy),
                 'auc': auc}

    return data_dict


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/evaluation.json')
    parser.add_argument('--vggface2', action='store_true')
    parser.add_argument('--lfw', action='store_true')
    parser.add_argument('--cox_video1', action='store_true')
    parser.add_argument('--cox_video2', action='store_true')
    parser.add_argument('--cox_video3', action='store_true')
    parser.add_argument('--cox_video4', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--all_cox', action='store_true')

    return parser.parse_args(argv)

if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])

    print('Feature extractor evaluation.')
    print('CONFIGURATION:\t{}'.format(args.config))
    with open(args.config) as json_config_file:
        config = utils.AttrDict(json.load(json_config_file))

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load model
    if config.model.checkpoint:
        checkpoint_path = config.model.checkpoint_path
    else:
        checkpoint_path = None
    model = models.load_model(config.model.model_arch,
                              device,
                              checkpoint_path=checkpoint_path,
                              embedding_size=config.model.embedding_size,
                              imgnet_pretrained=config.model.pretrained_imagenet)

    batch_size = (config.hyperparameters.people_per_batch * config.hyperparameters.images_per_person) // 2
    nrof_folds = config.dataset.cross_validation.num_fold

    fold_tool = utils.FoldGenerator(nrof_folds,
                                    config.dataset.cross_validation.num_train_folds,
                                    config.dataset.cross_validation.num_val_folds)
    train_folds, val_folds, test_folds = fold_tool.get_fold()

    #Data transform
    data_transform = transforms.Compose([
        transforms.Resize((config.hyperparameters.image_size, config.hyperparameters.image_size), interpolation=1),
        transforms.ToTensor()
    ])

    # Get data loaders
    if args.all:
        args.vggface2 = True
        args.lfw = True
        args.cox_video1 = True
        args.cox_video2 = True
        args.cox_video3 = True
        args.cox_video4 = True

    if args.all_cox:
        args.cox_video1 = True
        args.cox_video2 = True
        args.cox_video3 = True
        args.cox_video4 = True

    test_loaders_list = Get_TestDataloaders(config,
                                            data_transform,
                                            batch_size,
                                            test_folds,
                                            nrof_folds,
                                            is_vggface2=args.vggface2,
                                            is_lfw=args.lfw,
                                            is_cox_video1=args.cox_video1,
                                            is_cox_video2=args.cox_video2,
                                            is_cox_video3=args.cox_video3,
                                            is_cox_video4=args.cox_video4)

    if not test_loaders_list:
        print('No datasets selected for evaluation.')
        print('Evaluation terminated.')
        exit(0)

    # Launch evaluation
    for test_name, test_loader in test_loaders_list:

        print('\nEvaluation on {}'.format(test_name))
        Evaluate(test_loader,
                 model,
                 device,
                 0,
                 plotter=None,
                 name=test_name,
                 nrof_folds=nrof_folds,
                 distance_metric=0,
                 val_far=config.hyperparameters.val_far,
                 plot_distances=False)

