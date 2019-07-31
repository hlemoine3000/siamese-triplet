
import argparse
import configparser
import cv2
import os
import time
import sys
import signal
import json
import tqdm
from PIL import Image
from datetime import datetime
import numpy as np

import torch
from torchvision import transforms
from torch.utils import data
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.cluster import v_measure_score

import utils
from utils.video import Video_Reader
from utils.visualization import draw_bounding_box_on_image_array
from utils import plotter
from ml_utils import ml_utils, clustering
import models
from dataset_utils.dataset import ImageFolderTrackDataset
from vis_embeddings import TSNE_Visualizer
from copy import deepcopy


def switch_labels(prediction, groundtruth):
    pred = prediction.copy()
    # Fit prediction labels to the groundtruth labels
    for i in range(len(np.unique(prediction))):
        # Find index of the most predicted class
        conf_matrix = confusion_matrix(groundtruth, pred)
        max_idx = np.where(conf_matrix[i] == np.amax(conf_matrix[i]))[0][0]
        # Do not perform permutation on previous classes
        if max_idx > i:
            # Find the indexes to change with the correct label
            max_indexes_to_change = np.where(pred == max_idx)[0]
            actual_indexes_to_change = np.where(pred == i)[0]
            # Change the prediction label with the new label
            np.put(pred, max_indexes_to_change, i)
            np.put(pred, actual_indexes_to_change, max_idx)

    return pred


def compute_metrics(prediction, groundtruth):
    acc = accuracy_score(groundtruth, prediction)
    v_score = v_measure_score(groundtruth, prediction)
    print('Clustering accuracy: {}'.format(acc))
    print('V measure score: {}'.format(v_score))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/video_description.json')

    return parser.parse_args(argv)


if __name__ == '__main__':

    # ffmpeg - i bbtS01E01.mkv - vf scale = 1024:576 - r 25 - codec: a copy outputbbt1.mkv

    annotation_path = '/export/livia/data/lemoineh/CVPR2013_PersonID_data/bbt_s01e01_facetracks.mat'
    movie_path = '/export/livia/data/lemoineh/BBT/bbts01e01.mkv'
    # checkpoint_path = '/export/livia/data/lemoineh/torch_facetripletloss/models/BBTfinetune2/model_209.pth'
    video_out_dir = '/export/livia/data/lemoineh/video/'

    max_frame = 100000

    args = parse_arguments(sys.argv[1:])

    print('Feature extractor evaluation.')
    print('CONFIGURATION:\t{}'.format(args.config))
    with open(args.config) as json_config_file:
        config = utils.AttrDict(json.load(json_config_file))

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # device = torch.device("cpu")

    idx_to_classes = {0: 'sheldon',
                      1: 'leonard',
                      2: 'unknown',
                      3: 'penny',
                      4: 'raj',
                      5: 'howard',
                      6: 'kurt'}

    classes_to_idx = {'sheldon': 0,
                      'leonard': 1,
                      'unknown': 2,
                      'penny': 3,
                      'raj': 4,
                      'howard': 5,
                      'kurt': 6}

    # Data transform
    data_transform = transforms.Compose([
        transforms.Resize((config.hyperparameters.image_size, config.hyperparameters.image_size), interpolation=1),
        transforms.ToTensor()
    ])

    video_dataset = ImageFolderTrackDataset('/export/livia/data/lemoineh/BBT/ep01_2', transform=data_transform)
    video_dataloader = data.DataLoader(video_dataset,
                                       num_workers=2,
                                       batch_size=100,
                                       pin_memory=True)
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

    print('Extracting features.')
    features = []
    imagetrack_labels = []
    gt_labels = []
    model.eval()
    tbar = tqdm.tqdm(video_dataloader)

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

    # # Frame level clustering
    # print('Performing frame level clustering.')
    #
    # pred_labels_list, silhouette_scores, range_n_clusters = clustering.kmeans_silhouetteanalysis(features,
    #                                                                                              7,
    #                                                                                              return_all=True,
    #                                                                                              verbose=True)
    # best_n_clusters_idx = silhouette_scores.index(max(silhouette_scores))
    # pred_labels = pred_labels_list[best_n_clusters_idx]
    #
    # print('Silhouette score: {}'.format(silhouette_scores[best_n_clusters_idx]))
    # print('Number of clusters: {}'.format(range_n_clusters[best_n_clusters_idx]))
    #
    # cluster_classes = np.unique(pred_labels)
    #
    # # Fit prediction labels to the groundtruth labels
    # pred_labels = switch_labels(pred_labels, gt_labels)
    #
    # fl_conf_matrix = confusion_matrix(gt_labels, pred_labels)
    # compute_metrics(pred_labels, gt_labels)
    # print('Confusion matrix:')
    # print(fl_conf_matrix)

    # Track level clustering
    print('Performing track level clustering.')
    mean_feature_tracklist = []
    tracklabels = []
    gt_tracklabels = []
    track_classes = np.unique(imagetrack_labels)
    for track_class in track_classes:
        features_indexes = [i for i, e in enumerate(imagetrack_labels) if e == track_class]
        features_track = features[features_indexes]
        mean_feature_tracklist.append(np.mean(features_track, axis=0))
        tracklabels.append(track_class)
        gt_tracklabels.append(gt_labels[features_indexes[0]])
    mean_feature_tracklist = np.asarray(mean_feature_tracklist)

    pred_tracklabels_list, silhouette_scores, range_n_clusters = clustering.hac_silhouetteanalysis(
        mean_feature_tracklist,
        7,
        verbose=True)
    best_n_clusters_idx = silhouette_scores.index(max(silhouette_scores))
    for i in range(len(pred_tracklabels_list)):

        print('___________________')
        print('Number of clusters: {}'.format(range_n_clusters[i]))
        print('Silhouette score: {}'.format(silhouette_scores[i]))

        pred_tracklabels = pred_tracklabels_list[i]
        # Fit prediction labels to the groundtruth labels
        pred_tracklabels = switch_labels(pred_tracklabels, gt_tracklabels)

        tl_conf_matrix = confusion_matrix(gt_tracklabels, pred_tracklabels)
        compute_metrics(pred_tracklabels, gt_tracklabels)
        print('Confusion matrix:')
        print(tl_conf_matrix)

    # Visdom visualisation
    tsne_vis = TSNE_Visualizer()
    tsne_vis.train(mean_feature_tracklist)
    tsne_vis.plot(np.asarray(tracklabels), env_name='BBT_vis', name='Emb_visTrack')
    tsne_vis.plot(np.asarray(gt_tracklabels), env_name='BBT_vis', name='Emb_visGT')
    # tsne_vis.plot(pred_tracklabels,env_name='BBT_vis', name='Emb_visPred')
    for lbl, num_cluster in zip(pred_tracklabels_list, range_n_clusters):
        tsne_vis.plot(lbl,env_name='BBT_vis', name='Emb_visPred{}'.format(num_cluster))

    print('Process completed.')
