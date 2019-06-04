
import argparse
import sys
import numpy as np
import tqdm
import pandas as pd
import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from copy import deepcopy
import utils
from dataset_utils import get_vggface2_testset, get_coxs2v_testset, get_lfw_testset


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
        for images_batch, issame, path_batch in tbar:
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

    tpr, fpr, accuracy, best_threshold = utils.Calculate_Roc(thresholds, embeddings1, embeddings2,
                                                             np.asarray(issame_array), nrof_folds=nrof_folds,
                                                             distance_metric=distance_metric,
                                                             subtract_mean=subtract_mean)

    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far, threshold_lowfar = utils.Calculate_Val(thresholds, embeddings1, embeddings2,
                                                              np.asarray(issame_array),
                                                              val_far,
                                                              nrof_folds=nrof_folds,
                                                              distance_metric=distance_metric,
                                                              subtract_mean=subtract_mean)

    print('Accuracy: {:.3%}+-{:.3%}'.format(np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: {:.3%}+-{:.3%} @ FAR={:.3%}'.format(val, val_std, far))
    print('Best threshold: {:.3f}'.format(best_threshold))

    if plotter:
        if plot_distances:
            plotter.plot('distance', 'step', name + '_an', 'Pairwise mean distance', step, negative_mean_distance)
            plotter.plot('distance', 'step', name + '_ap', 'Pairwise mean distance', step, positive_mean_distance)

        plotter.plot('accuracy', 'step', name, 'Accuracy', step, np.mean(accuracy))
        plotter.plot('validation rate', 'step', name, 'Validation Rate @ FAR={:.3%}'.format(val_far), step, val)
        plotter.plot('best threshold', 'step', name, 'Best Threshold', step, best_threshold)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/da_config.json')
    parser.add_argument('--vggface2', action='store_true')
    parser.add_argument('--lfw', action='store_true')
    parser.add_argument('--cox_video1', action='store_true')
    parser.add_argument('--cox_video2', action='store_true')
    parser.add_argument('--cox_video3', action='store_true')
    parser.add_argument('--cox_video4', action='store_true')

    return parser.parse_args(argv)

# if __name__ == '__main__':
#
#     args = parse_arguments(sys.argv[1:])
#
#     print('Feature extractor evaluation.')
#     print('CONFIGURATION:\t{}'.format(args.config))
#     with open(args.config) as json_config_file:
#         config = utils.AttrDict(json.load(json_config_file))
#
#     test_loaders_list = []
#     batch_size = (config.hyperparameters.people_per_batch * config.hyperparameters.images_per_person) // 2
#     nrof_folds = config.dataset.cross_validation.num_fold
#
#     #Data transform
#     data_transform = transforms.Compose([
#         transforms.Resize((config.hyperparameters.image_size, config.hyperparameters.image_size), interpolation=1),
#         transforms.ToTensor()
#     ])
#
#     # Get data loaders
#     if args.vggface2:
#         data_loader = get_vggface2_testset(config.dataset.vggface2.test_dir,
#                                            config.dataset.vggface2.pairs_file,
#                                            data_transform,
#                                            batch_size)
#         test_loaders_list.append(('vggface2', data_loader))
#
#     if args.lfw:
#         test_loaders_list.append(get_lfw_testset(config.dataset.lfw.test_dir,
#                                                  config.dataset.lfw.pairs_file,
#                                                  data_transform,
#                                                  batch_size))
#         test_loaders_list.append(('lfw', data_loader))
#
#     for test_name, test_loader in test_loaders_list:
#
#         Evaluate(test_loader,
#                  model,
#                  device,
#                  step,
#                  plotter=None,
#                  name='evaluation',
#                  nrof_folds=10,
#                  distance_metric=0,
#                  val_far=1e-3,
#                  plot_distances=False)

