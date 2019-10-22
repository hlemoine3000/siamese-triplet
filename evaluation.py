
import argparse
import sys
import numpy as np
import tqdm
import pandas as pd
import json
from sklearn import metrics
from sklearn import cluster
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
from torch import nn
from torchvision import transforms

from copy import deepcopy
import utils
import models
import dataset_utils
from utils import plotter_utils
from ml_utils import clustering


class Evaluator():
    def __init__(self,
                 test_loader: torch.utils.data.DataLoader,
                 test_name: str):

        self.test_name = test_name
        self.test_loader = test_loader

    def evaluate(self,
                 model: nn.Module,
                 device,
                 plotter: plotter_utils.VisdomPlotter = None,
                 epoch: int = 0):
        pass


class Video_Description_Evaluator(Evaluator):
    def __init__(self,
                 test_loader: torch.utils.data.DataLoader,
                 test_name: str):
        super().__init__(test_loader, test_name)

    def evaluate(self,
                 model: nn.Module,
                 device,
                 plotter: plotter_utils.VisdomPlotter = None,
                 epoch: int = 0):

        print('Extracting features.')
        features = []
        imagetrack_labels = []
        gt_labels = []

        model.eval()
        tbar = tqdm.tqdm(self.test_loader)
        with torch.no_grad():
            for images_batch, track_labels_batch, gt_labels_batch in tbar:
                # Forward pass
                images_batch = images_batch.to(device)
                features.append(model.forward(images_batch))
                imagetrack_labels.append(deepcopy(track_labels_batch))
                gt_labels.append(deepcopy(gt_labels_batch))

        features = torch.cat(features, 0).cpu().numpy()
        imagetrack_labels = torch.cat(imagetrack_labels, 0).cpu().numpy()
        gt_labels = torch.cat(gt_labels, 0).cpu().numpy()

        # Track level clustering
        print('Performing track level clustering.')
        mean_feature_tracklist = []
        tracklabels = []
        gt_tracklabels = []
        track_classes = np.unique(imagetrack_labels)
        # Compute mean of each track
        print('Compute mean representation for each track.')
        for track_class in track_classes:
            features_indexes = [i for i, e in enumerate(imagetrack_labels) if e == track_class]
            features_track = features[features_indexes]
            mean_feature_tracklist.append(np.mean(features_track, axis=0))
            tracklabels.append(track_class)
            gt_tracklabels.append(gt_labels[features_indexes[0]])
        mean_feature_tracklist = np.asarray(mean_feature_tracklist)
        gt_tracklabels = np.asarray(gt_tracklabels)

        clustering.evaluate_clustering(mean_feature_tracklist,
                                       gt_tracklabels,
                                       ['kmeans', 'hac', 'spectral'],
                                       plotter=plotter,
                                       epoch=epoch)


class Clustering_Evaluator(Evaluator):
    def __init__(self,
                 test_loader: torch.utils.data.DataLoader,
                 test_name: str):
        super().__init__(test_loader, test_name)

    def evaluate(self,
                 model,
                 device,
                 plotter: plotter_utils.VisdomPlotter = None,
                 epoch: int = 0):
        """Evaluation for target encoder by source classifier on target dataset."""
        # set eval state for Dropout and BN layers
        features = []
        labels = []
        model.eval()
        tbar = tqdm.tqdm(self.test_loader)
        with torch.no_grad():
            for images_batch, labels_batch in tbar:
                # Forward pass
                images_batch = images_batch.to(device)
                features.append(model.forward(images_batch))
                labels.append(deepcopy(labels_batch))

        features = torch.cat(features, 0).cpu().numpy()
        labels = torch.cat(labels, 0).cpu().numpy()

        # Kmeans classification
        clusterer = cluster.KMeans(n_clusters=10, random_state=10)
        pred_labels = clusterer.fit_predict(features)

        purity = clustering.purity_score(labels, pred_labels)

        print('Clustering purity: {:.3}'.format(purity))
        plotter.plot('metrics value', 'epoch', '{}_purity'.format(self.test_name), 'Clustering Performance', epoch,
                     purity)

        # Embeddings visualisation
        tsne_features = TSNE(n_components=2,
                             perplexity=30.0,
                             early_exaggeration=12.0, learning_rate=200.0, n_iter=20000,
                             n_iter_without_progress=1000, min_grad_norm=1e-7,
                             metric="euclidean", init="random", verbose=1,
                             random_state=None, method='exact', angle=0.2).fit_transform(features[0:200])

        if epoch == 0:
            title_gt = '{} Embeddings Ground Truth Epoch 0'.format(self.test_name)
            title_prd = '{} Predictions Epoch 0'.format(self.test_name)
        else:
            title_gt = '{} Embeddings Ground Truth'.format(self.test_name)
            title_prd = '{} Predictions'.format(self.test_name)

        plotter.scatter_plot(title_gt,
                             tsne_features,
                             labels[0:200])
        plotter.scatter_plot(title_prd,
                             tsne_features,
                             pred_labels[0:200])


class Pairs_Evaluator(Evaluator):
    def __init__(self,
                 test_loader: torch.utils.data.DataLoader,
                 test_name: str,
                 nrof_folds=10,
                 distance_metric=0,
                 val_far=1e-3,
                 subtract_mean=False):
        super().__init__(test_loader, test_name)

        self.nrof_folds = nrof_folds
        self.distance_metric = distance_metric
        self.val_far = val_far
        self.subtract_mean = subtract_mean

    def evaluate(self,
                 model,
                 device,
                 plotter: plotter_utils.VisdomPlotter = None,
                 epoch: int = 0):
        embeddings1 = []
        embeddings2 = []
        issame_array = []

        model.eval()

        with torch.no_grad():
            tbar = tqdm.tqdm(self.test_loader, dynamic_ncols=True)
            for images_batch, issame in tbar:
                # Transfer to GPU
                image_batch = torch.cat(images_batch, 0).to(device)

                emb = model.forward(image_batch)
                emb1, emb2 = torch.chunk(emb, 2, 0)

                embeddings1.append(emb1)
                embeddings2.append(emb2)
                issame_array.append(deepcopy(issame))

            embeddings1 = torch.cat(embeddings1, 0).cpu().numpy()
            embeddings2 = torch.cat(embeddings2, 0).cpu().numpy()
            issame_array = torch.cat(issame_array, 0).cpu().numpy()

        distance_and_is_same = zip(np.sum((embeddings1 - embeddings2) ** 2, axis=1), issame_array)
        distance_and_is_same_df = pd.DataFrame(distance_and_is_same)
        negative_distances = distance_and_is_same_df[distance_and_is_same_df[1] == False][0]
        positive_distances = distance_and_is_same_df[distance_and_is_same_df[1] == True][0]
        negative_mean_distance = negative_distances.mean()
        positive_mean_distance = positive_distances.mean()

        thresholds = np.arange(0, 15, 0.01)
        subtract_mean = False

        tpr, fpr, accuracy = utils.Calculate_Roc(thresholds, embeddings1, embeddings2,
                                                 np.asarray(issame_array), nrof_folds=self.nrof_folds,
                                                 distance_metric=self.distance_metric,
                                                 subtract_mean=subtract_mean)

        thresholds = np.arange(0, 15, 0.001)
        val, val_std, far, threshold_lowfar = utils.Calculate_Val(thresholds, embeddings1, embeddings2,
                                                                  np.asarray(issame_array),
                                                                  self.val_far,
                                                                  nrof_folds=self.nrof_folds,
                                                                  distance_metric=self.distance_metric,
                                                                  subtract_mean=subtract_mean)

        print('Accuracy: {:.3%}+-{:.3%}'.format(np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: {:.3%}+-{:.3%} @ FAR={:.3%}'.format(val, val_std, far))
        print('Positive mean distances: {:.3}'.format(positive_mean_distance))
        print('Negative mean distances: {:.3}'.format(negative_mean_distance))

        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.5f' % auc)

        if plotter:
            plotter.plot('distance', 'step', '{}_an'.format(self.test_name), 'Pairwise mean distance', epoch,
                         negative_mean_distance)
            plotter.plot('distance', 'step', '{}_ap'.format(self.test_name), 'Pairwise mean distance', epoch,
                         positive_mean_distance)

            plotter.plot('accuracy', 'epoch', self.test_name, 'Accuracy', epoch, np.mean(accuracy))
            plotter.plot('auc', 'epoch', self.test_name, 'AUC', epoch, auc)
            plotter.plot('validation rate', 'step', self.test_name, 'Validation Rate @ FAR={:.3%}'.format(self.val_far), epoch,
                         val)

            step = 0.05
            max_distance = max(max(negative_distances), max(positive_distances)) + step
            bins = np.arange(0.0, max_distance, step)
            negative_hist, _ = np.histogram(negative_distances, bins=bins)
            positive_hist, _ = np.histogram(positive_distances, bins=bins)
            hist = np.column_stack((positive_hist, negative_hist))

            title = '{} Distances Distribution Epoch 0'.format(self.test_name)
            if not plotter.plot_exist(title):
                title = '{} Distances Distribution'.format(self.test_name)

            if bins.shape[0] != hist.shape[1]:
                bins = bins[1:]
            plotter.stem_plot(title, 'Number of Samples',
                              '{} Distances at epoch {}'.format(self.test_name, epoch), ['pos', 'neg'], bins, hist)



        data_dict = {'accuracy': np.mean(accuracy),
                     'auc': auc}

        return data_dict

def get_eval_function(function_name):
    if function_name == 'pair_evaluation':
        return pair_evaluate
    else:
        raise Exception('Evaluation function {} does not exist.'.format(function_name))

def kmeans_evaluation(test_loader: torch.utils.data.DataLoader,
                            model,
                            device,
                            test_name: str,
                            plotter: plotter_utils.VisdomPlotter = None,
                            epoch: int = 0,
                            nrof_folds=10,
                            distance_metric=0,
                            val_far=1e-3):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    features = []
    labels = []
    model.eval()
    tbar = tqdm.tqdm(test_loader)
    with torch.no_grad():
        for images_batch, labels_batch in tbar:
            # Forward pass
            images_batch = images_batch.to(device)
            features.append(model.forward(images_batch))
            labels.append(deepcopy(labels_batch))

    features = torch.cat(features, 0).cpu().numpy()
    labels = torch.cat(labels, 0).cpu().numpy()

    # Kmeans classification
    clusterer = cluster.KMeans(n_clusters=10, random_state=10)
    pred_labels = clusterer.fit_predict(features)

    purity = clustering.purity_score(labels, pred_labels)

    print('Clustering purity: {:.3}'.format(purity))
    plotter.plot('metrics value', 'epoch', '{}_purity'.format(test_name), 'Clustering Performance', epoch,
                 purity)

    # Embeddings visualisation
    tsne_features = TSNE(n_components=2,
                         perplexity=30.0,
                         early_exaggeration=12.0, learning_rate=200.0, n_iter=20000,
                         n_iter_without_progress=1000, min_grad_norm=1e-7,
                         metric="euclidean", init="random", verbose=1,
                         random_state=None, method='exact', angle=0.2).fit_transform(features[0:200])

    if epoch == 0:
        title_gt = '{} Embeddings Ground Truth Epoch 0'.format(test_name)
        title_prd = '{} Predictions Epoch 0'.format(test_name)
    else:
        title_gt = '{} Embeddings Ground Truth'.format(test_name)
        title_prd = '{} Predictions'.format(test_name)

    plotter.scatter_plot(title_gt,
                         tsne_features,
                         labels[0:200])
    plotter.scatter_plot(title_prd,
                         tsne_features,
                         pred_labels[0:200])


def video_description_evaluate(test_loader: torch.utils.data.DataLoader,
                               model,
                               device,
                               test_name: str,
                               plotter: plotter_utils.VisdomPlotter = None,
                               epoch: int = 0,
                               nrof_folds=10,
                               distance_metric=0,
                               val_far=1e-3):

    print('Extracting features.')
    features = []
    imagetrack_labels = []
    gt_labels = []

    model.eval()
    tbar = tqdm.tqdm(test_loader)
    with torch.no_grad():
        for images_batch, track_labels_batch, gt_labels_batch in tbar:
            # Forward pass
            images_batch = images_batch.to(device)
            features.append(model.forward(images_batch))
            imagetrack_labels.append(deepcopy(track_labels_batch))
            gt_labels.append(deepcopy(gt_labels_batch))

    features = torch.cat(features, 0).cpu().numpy()
    imagetrack_labels = torch.cat(imagetrack_labels, 0).cpu().numpy()
    gt_labels = torch.cat(gt_labels, 0).cpu().numpy()

    # Track level clustering
    print('Performing track level clustering.')
    mean_feature_tracklist = []
    tracklabels = []
    gt_tracklabels = []
    track_classes = np.unique(imagetrack_labels)
    # Compute mean of each track
    print('Compute mean representation for each track.')
    for track_class in track_classes:
        features_indexes = [i for i, e in enumerate(imagetrack_labels) if e == track_class]
        features_track = features[features_indexes]
        mean_feature_tracklist.append(np.mean(features_track, axis=0))
        tracklabels.append(track_class)
        gt_tracklabels.append(gt_labels[features_indexes[0]])
    mean_feature_tracklist = np.asarray(mean_feature_tracklist)
    gt_tracklabels = np.asarray(gt_tracklabels)

    clustering.evaluate_clustering(mean_feature_tracklist,
                                   gt_tracklabels,
                                   ['kmeans', 'hac', 'spectral'],
                                   plotter=plotter,
                                   epoch=epoch)


def pair_evaluate(test_loader: torch.utils.data.DataLoader,
                  model,
                  device,
                  test_name: str,
                  plotter: plotter_utils.VisdomPlotter=None,
                  epoch: int=0,
                  nrof_folds=10,
                  distance_metric=0,
                  val_far=1e-3):

    embeddings1 = []
    embeddings2 = []
    issame_array = []

    model.eval()

    with torch.no_grad():
        tbar = tqdm.tqdm(test_loader, dynamic_ncols=True)
        for images_batch, issame in tbar:
            # Transfer to GPU
            image_batch = torch.cat(images_batch,0).to(device)

            emb = model.forward(image_batch)
            emb1, emb2 = torch.chunk(emb, 2, 0)

            embeddings1.append(emb1)
            embeddings2.append(emb2)
            issame_array.append(deepcopy(issame))

        embeddings1 = torch.cat(embeddings1, 0).cpu().numpy()
        embeddings2 = torch.cat(embeddings2, 0).cpu().numpy()
        issame_array = torch.cat(issame_array, 0).cpu().numpy()

    distance_and_is_same = zip(np.sum((embeddings1 - embeddings2) ** 2, axis=1), issame_array)
    distance_and_is_same_df = pd.DataFrame(distance_and_is_same)
    negative_distances = distance_and_is_same_df[distance_and_is_same_df[1] == False][0]
    positive_distances = distance_and_is_same_df[distance_and_is_same_df[1] == True][0]
    negative_mean_distance = negative_distances.mean()
    positive_mean_distance = positive_distances.mean()

    thresholds = np.arange(0, 15, 0.01)
    subtract_mean = False

    tpr, fpr, accuracy = utils.Calculate_Roc(thresholds, embeddings1, embeddings2,
                                                             np.asarray(issame_array), nrof_folds=nrof_folds,
                                                             distance_metric=distance_metric,
                                                             subtract_mean=subtract_mean)

    thresholds = np.arange(0, 15, 0.001)
    val, val_std, far, threshold_lowfar = utils.Calculate_Val(thresholds, embeddings1, embeddings2,
                                                              np.asarray(issame_array),
                                                              val_far,
                                                              nrof_folds=nrof_folds,
                                                              distance_metric=distance_metric,
                                                              subtract_mean=subtract_mean)

    print('Accuracy: {:.3%}+-{:.3%}'.format(np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: {:.3%}+-{:.3%} @ FAR={:.3%}'.format(val, val_std, far))
    print('Positive mean distances: {:.3}'.format(positive_mean_distance))
    print('Negative mean distances: {:.3}'.format(negative_mean_distance))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.5f' % auc)

    if plotter:
        plotter.plot('distance', 'step', '{}_an'.format(test_name), 'Pairwise mean distance', epoch, negative_mean_distance)
        plotter.plot('distance', 'step', '{}_ap'.format(test_name), 'Pairwise mean distance', epoch, positive_mean_distance)

        plotter.plot('accuracy', 'epoch', test_name, 'Accuracy', epoch, np.mean(accuracy))
        plotter.plot('auc', 'epoch', test_name, 'AUC', epoch, auc)
        plotter.plot('validation rate', 'step', test_name, 'Validation Rate @ FAR={:.3%}'.format(val_far), epoch, val)

        step = 0.05
        max_distance = max(max(negative_distances), max(positive_distances)) + step
        bins = np.arange(0.0, max_distance, step)
        negative_hist, _ = np.histogram(negative_distances, bins=bins)
        positive_hist, _ = np.histogram(positive_distances, bins=bins)
        hist = np.column_stack((positive_hist, negative_hist))

        epoch0_title = '{} Distances Distribution Epoch 0'.format(test_name)
        title = '{} Distances Distribution'.format(test_name)
        if not plotter.plot_exist(epoch0_title):
            plotter.stem_plot(epoch0_title, 'Number of Samples', '{} Distances at epoch {}'.format(test_name, epoch), ['pos', 'neg'], bins[1:], hist)
        else:
            plotter.stem_plot(title, 'Number of Samples', '{} Distances at epoch {}'.format(test_name, epoch), ['pos', 'neg'], bins[1:], hist)

    data_dict = {'accuracy': np.mean(accuracy),
                 'auc': auc}

    return data_dict


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/evaluation.json')
    parser.add_argument('--experiment', type=str,
                        help='Dataset to experiment', default='lfw')

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
    test_loaders_list = dataset_utils.dataloaders.get_testevaluators(config,
                                                                     data_transform,
                                                                     batch_size,
                                                                     test_folds,
                                                                     nrof_folds,
                                                                     ['bbt_ep01'])

    plotter = utils.VisdomPlotter(config.visdom.server,
                                  env_name='evaluation',
                                  port=config.visdom.port)

    if not test_loaders_list:
        print('No datasets selected for evaluation.')
        print('Evaluation terminated.')
        exit(0)

    # Launch evaluation
    for test_name, test_loader, eval_function in test_loaders_list:
        print('\nEvaluation on {}'.format(test_name))
        eval_function(test_loader,
                      model,
                      device,
                      test_name,
                      plotter=plotter,
                      epoch=0,
                      nrof_folds=nrof_folds,
                      distance_metric=0,
                      val_far=config.hyperparameters.val_far)

