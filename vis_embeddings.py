
import argparse
import sys
import numpy as np
import tqdm
import pandas as pd
import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from sklearn.manifold import TSNE

import utils
import models
from dataset_utils.dataloaders import Get_TestDataloaders


def vis_embeddings():
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_embedded = TSNE(n_components=2).fit_transform(X)
    X_embedded.shape


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
                                            is_vggface2=args.vggface2,
                                            is_lfw=args.lfw,
                                            is_cox_video1=args.cox_video1,
                                            is_cox_video2=args.cox_video2,
                                            is_cox_video3=args.cox_video3,
                                            is_cox_video4=args.cox_video4)

    # Load model
    print('Loading from checkpoint {}'.format(config.model.checkpoint_path))
    checkpoint = torch.load(config.model.checkpoint_path)
    embedding_size = checkpoint['embedding_size']

    model = models.load_model(config.model.model_arch,
                              embedding_size=embedding_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

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