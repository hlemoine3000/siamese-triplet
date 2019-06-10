
import argparse
import sys
import numpy as np
import tqdm
import pandas as pd
import json
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from sklearn.manifold import TSNE

import utils
import models
from dataset_utils import dataloaders
from utils.plotter import VisdomScatterPlotter


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/vis_config.json')
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

    nrof_folds = config.dataset.cross_validation.num_fold
    fold_tool = utils.FoldGenerator(nrof_folds,
                                    config.dataset.cross_validation.num_train_folds,
                                    config.dataset.cross_validation.num_val_folds)
    train_folds, val_folds, test_folds = fold_tool.get_fold()

    # Data transform
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

    # Load model
    print('Loading from checkpoint {}'.format(config.model.checkpoint_path))
    checkpoint = torch.load(config.model.checkpoint_path)
    embedding_size = checkpoint['embedding_size']

    model = models.load_model(config.model.model_arch,
                              embedding_size=embedding_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    data_loaders_list = dataloaders.Get_TrainDataloaders(config,
                                                         data_transform,
                                                         config.embeddings_visualisation.num_class,
                                                         config.embeddings_visualisation.num_sample_per_class,
                                                         [0],
                                                         nrof_folds,
                                                         is_vggface2=False,
                                                         is_cox_video1=False,
                                                         is_cox_video2=True,
                                                         is_cox_video3=False,
                                                         is_cox_video4=False)

    plotter = utils.VisdomLinePlotter(env_name=config.visdom.environment_name, port=config.visdom.port)

    # Launch embeddings extraction
    embeddings = []
    label_array = []
    textlabels = []

    tbar = tqdm.tqdm(data_loaders_list)
    with torch.no_grad():
        for name, data_loader in tbar:

            print('\nExtracting embeddings on {}'.format(name))

            #extract one batch of each datasets
            images_batch, label_batch = next(iter(data_loader))

            # Transfer to GPU
            image_batch = images_batch.to(device, non_blocking=True)

            emb = model.forward(image_batch)

            embeddings.append(emb)
            label_array.append(deepcopy(label_batch))

            embeddings = torch.cat(embeddings, 0).cpu().numpy()
            label_array = torch.cat(label_array, 0).cpu().numpy()

            # test label in dataset
            for label in label_array:
                for class_name, idx in data_loader.dataset.class_to_idx.items():
                    if idx == label:
                        textlabels.append(class_name)

    X_embedded = TSNE(n_components=2, perplexity=20.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=10000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=1,
                 random_state=None, method='exact', angle=0.2).fit_transform(embeddings)
    legends = utils.unique(textlabels)

    my_plot = VisdomScatterPlotter(env_name=config.visdom.environment_name,
                                   port=config.visdom.port)

    label_array += 1
    my_plot.plot('Embeddings Visualisation TSNE',
                 X_embedded,
                 label_array,
                 legends=legends)

    print('Visualisation completed.')