
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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.decomposition import PCA

import utils
from utils.video import Video_Reader
from utils.visualization import draw_bounding_box_on_image_array
from utils import plotter
from ml_utils import ml_utils, clustering
import models
from dataset_utils.bbt import Read_Annotation
from dataset_utils.dataset import NumpyDataset
from vis_embeddings import TSNE_Visualizer

images_path = 'images/'
database_output_path = 'database/'


def switch_labels(prediction, groundtruth):
    pred = prediction.copy()
    # Fit prediction labels to the groundtruth labels
    for i in range(len(cluster_classes)):
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


def signal_handler(sig, frame):
    if out:
        out.release()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


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

    # Create video source instance
    print('Initializing video capture at {}'.format(movie_path))
    video_src = Video_Reader(movie_path)

    if not os.path.exists(video_out_dir):
        os.mkdir(video_out_dir)

    _, image = video_src.get_frame()

    img_height, img_width, img_channel = image.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filename = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    video_out_filepath = video_out_dir + filename + '.avi'
    out = cv2.VideoWriter(video_out_filepath, fourcc, 25, (img_width, img_height))

    print('Reading annotation at {}'.format(annotation_path))
    Annotation_list = Read_Annotation(annotation_path, (img_width, img_height))

    cropped_image_list = []
    track_dict = {}
    gt_tracklabels_list =[]
    frame_dict = {}
    gt_labels_dict = {}
    gt_labels_list = []
    gt_idx_list = []
    num_frame = min(len(Annotation_list), max_frame)

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


    print('Extracting face patches.')

    video_src.reset()
    frame_idx = 0
    image_idx = 0
    tbar = tqdm.tqdm(range(num_frame))
    for j in tbar:

        ret, image = video_src.get_frame()
        if not ret:
            break

        frame_annotations = Annotation_list[frame_idx]

        for annotation in frame_annotations:

            cropped_image = image[annotation[1]: annotation[3], annotation[0]: annotation[2],:]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            # cropped_image = np.transpose(cropped_image, (2, 0, 1))
            cropped_image = Image.fromarray(cropped_image)
            cropped_image_list.append(cropped_image)

            # # Grondtruth labels
            # if annotation[4] not in classes_to_idx.keys():
            #     # Add new groundtruth label
            #     classes_to_idx[annotation[4]] = num_gt_classes
            #     idx_to_classes[num_gt_classes] = annotation[4]
            #     gt_idx_list.append(num_gt_classes)
            #     num_gt_classes += 1

            # Add image index toactual track
            if annotation[6] not in track_dict.keys():
                # Init Image Indexes list for the new track
                gt_tracklabels_list.append(classes_to_idx[annotation[4]])
                track_dict[annotation[6]] = [image_idx]
            else:
                track_dict[annotation[6]].append(image_idx)

            gt_labels_list.append(classes_to_idx[annotation[4]])

            image_idx += 1

        frame_idx += 1
    print('')

    # Data transform
    data_transform = transforms.Compose([
        transforms.Resize((config.hyperparameters.image_size, config.hyperparameters.image_size), interpolation=1),
        transforms.ToTensor()
    ])

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

    dataset = NumpyDataset(cropped_image_list,
                           transform=data_transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             num_workers=2,
                                             batch_size=100,
                                             pin_memory=True)

    print('Extracting features.')
    features = ml_utils.extract_features(dataloader,
                                         model,
                                         device)

    # Frame level clustering
    # print('Performing frame level clustering.')
    #
    # pred_labels_list, silhouette_scores, range_n_clusters = clustering.hac_silhouetteanalysis(features,
    #                                                                                   7,
    #                                                                                   verbose=True)
    # best_n_clusters_idx = silhouette_scores.index(max(silhouette_scores))
    # pred_labels = pred_labels_list[best_n_clusters_idx]
    #
    # print('Silhouette score: {}'.format(silhouette_scores[best_n_clusters_idx]))
    # print('Number of clusters: {}'.format(range_n_clusters[best_n_clusters_idx]))
    #
    # cluster_classes = np.unique(pred_labels)
    # gt_labels = np.asarray(gt_labels_list)
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
    for track_idx in track_dict.keys():
        feature_track = features[track_dict[track_idx]]
        mean_feature_tracklist.append(np.mean(feature_track, axis=0))
    mean_feature_tracklist = np.asarray(mean_feature_tracklist)

    pred_tracklabels_list, silhouette_scores, range_n_clusters = clustering.kmeans_silhouetteanalysis(
        mean_feature_tracklist,
        7,
        return_all=True,
        verbose=True)
    best_n_clusters_idx = silhouette_scores.index(max(silhouette_scores))
    pred_tracklabels = pred_tracklabels_list[best_n_clusters_idx]

    print('Silhouette score: {}'.format(silhouette_scores[best_n_clusters_idx]))
    print('Number of clusters: {}'.format(range_n_clusters[best_n_clusters_idx]))

    cluster_classes = np.unique(pred_tracklabels)
    gt_tracklabels = np.asarray(gt_tracklabels_list)

    # Fit prediction labels to the groundtruth labels
    pred_tracklabels = switch_labels(pred_tracklabels, gt_tracklabels)

    tl_conf_matrix = confusion_matrix(gt_tracklabels, pred_tracklabels)
    compute_metrics(pred_tracklabels, gt_tracklabels)
    print('Confusion matrix:')
    print(tl_conf_matrix)

    # Visdom visualisation
    tsne_vis = TSNE_Visualizer()
    tsne_vis.train(mean_feature_tracklist)
    tsne_vis.plot(gt_tracklabels,env_name='BBT_vis', name='Emb_visGT')
    for lbl, num_cluster in zip(pred_tracklabels_list, range_n_clusters):
        tsne_vis.plot(lbl,env_name='BBT_vis', name='Emb_visPred{}'.format(num_cluster))

    # classes_list = list(classes_to_idx.keys())
    # heatmap_vis = plotter.VisdomHeatmap(env_name='BBT_vis')
    # heatmap_vis.plot('Confusion Matrix Frame Level', fl_conf_matrix, classes_list)
    # heatmap_vis.plot('Confusion Matrix Track Level', tl_conf_matrix, classes_list)

    # print('Describing video sequence.')
    #
    # video_src.reset()
    # frame_idx = 0
    # image_idx = 0
    #
    # tbar = tqdm.tqdm(range(num_frame))
    # for j in tbar:
    #
    #     ret, image = video_src.get_frame()
    #     if not ret:
    #         break
    #
    #     frame_annotations = Annotation_list[frame_idx]
    #
    #     for annotation in frame_annotations:
    #
    #         gt_label = idx_to_classes[gt_tracklabels_list[annotation[6]]]
    #         frame_label = idx_to_classes[pred_labels[image_idx]]
    #         track_label = idx_to_classes[pred_tracklabels[annotation[6]]]
    #
    #         if gt_label != frame_label and gt_label != track_label:
    #             color = '#ff5050'
    #         elif gt_label != track_label:
    #             color = '#ffff00'
    #         else:
    #             color = '#66ff66'
    #
    #         annotation_str = ['GT   :{}'.format(gt_label),
    #                           'Frame: {}'.format(frame_label),
    #                           'Track: {}'.format(track_label)]
    #
    #         draw_bounding_box_on_image_array(
    #             image,
    #             annotation[1],
    #             annotation[0],
    #             annotation[3],
    #             annotation[2],
    #             color=color,
    #             thickness=4,
    #             display_str_list=annotation_str,
    #             use_normalized_coordinates=False)
    #
    #         image_idx += 1
    #
    #     cv2.putText(image, "Frame {}".format(frame_idx), (10, 20),
    #                 cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
    #     out.write(image)
    #
    #     frame_idx += 1
    # print('')
    #
    # print('Saving video at ' + video_out_filepath)
    # out.release()

    print('Process completed.')
