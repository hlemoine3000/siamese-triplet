
import cv2
import os
import tqdm
from PIL import Image
import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
from torchvision import transforms
from torchvision.transforms.transforms import F
from torch import nn

import utils
from utils.fast_dt import FAST_DT
from utils.video import Video_Reader
from utils.visualization import draw_bounding_box_on_image_array
from ml_utils import ml_utils, clustering
from dataset_utils.dataset import NumpyDataset
from utils import projection_utils

color = {0: '#000000',
         1: '#FF0000',
         2: '#00FF00',
         3: '#0000FF',
         4: '#FFFF00',
         5: '#00FFFF',
         6: '#FF00FF',
         7: '#800000',
         8: '#808000',
         9: '#e4ab2b',
         10: '#008ff8'}


def get_bounding_boxes(movie_path: str,
                       max_frame: int=100000,
                       tracker_max_age: int=10):

    # Create video source instance
    print('Initializing video capture at {}'.format(movie_path))
    video_src = Video_Reader(movie_path)

    my_fastdt = FAST_DT(tracker_max_age=tracker_max_age)

    print('Extracting face patches.')

    bounding_boxes_list = []
    bbx_idx = 0
    tbar = tqdm.tqdm(range(max_frame))
    for frame_idx in tbar:

        ret, image = video_src.get_frame()
        if not ret:
            break

        bounding_boxes = my_fastdt.predict(image)
        for bbx in bounding_boxes:
            track_id = bbx[4]
            bounding_boxes_list.append([frame_idx, track_id, bbx_idx, bbx[1], bbx[3], bbx[0], bbx[2]])
            bbx_idx += 1

    return bounding_boxes_list


def get_cropped_images(movie_path: str,
                       bounding_boxes_list: list,
                       max_frame: int=100000,):

    bounding_boxes_dict = {}
    for bounding_box in bounding_boxes_list:
        frame_idx = bounding_box[0]
        if frame_idx not in bounding_boxes_dict.keys():
            bounding_boxes_dict[frame_idx] = [bounding_box[3:7]]
        else:
            bounding_boxes_dict[frame_idx].append(bounding_box[3:7])

    # Create video source instance
    video_src = Video_Reader(movie_path)

    cropped_image_list = []
    tbar = tqdm.tqdm(range(max_frame))
    for frame_idx in tbar:

        ret, image = video_src.get_frame()
        if not ret:
            break

        if frame_idx in bounding_boxes_dict.keys():

            bounding_boxes = bounding_boxes_dict[frame_idx]
            for bbx in bounding_boxes:
                cropped_image = image[bbx[1]: bbx[3], bbx[0]: bbx[2], :]
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                cropped_image = Image.fromarray(cropped_image)
                cropped_image = utils.make_square(cropped_image)
                cropped_image = F.resize(cropped_image, size=160, interpolation=1)
                cropped_image_list.append(cropped_image)

    return cropped_image_list


def get_bbx_dict(bounding_box_list: list) -> dict:

    bbx_dict = {}
    for bounding_box in bounding_box_list:
        bbx_idx = bounding_box[2]
        track_id = bounding_box[1]
        if bbx_idx not in bbx_dict.keys():
            bbx_dict[bbx_idx] = [track_id]
        else:
            bbx_dict[bbx_idx].append(track_id)

    return bbx_dict


def get_frame_dict(bounding_box_list: list) -> dict:

    frame_dict = {}
    for bounding_box in bounding_box_list:
        frame_idx = bounding_box[0]
        if frame_idx not in frame_dict.keys():
            frame_dict[frame_idx] = [bounding_box[3:]]
        else:
            frame_dict[frame_idx].append(bounding_box[3:])

    return frame_dict


def get_track_dict(bounding_box_list: list) -> dict:

    track_dict = {}
    for bbx_idx, bounding_box in enumerate(bounding_box_list):
        track_id = bounding_box[1]
        if track_id not in track_dict.keys():
            track_dict[track_id] = [bbx_idx]
        else:
            track_dict[track_id].append(bbx_idx)

    return track_dict


def write_video(movie_path: str,
                output_path: str,
                pred_labels: np.array,
                frame_dict: dict,
                name: str = 'video_out',
                max_frame: int = 100000):

    # Set up video reader
    video_src = Video_Reader(movie_path)
    _, image = video_src.get_frame()
    img_height, img_width, img_channel = image.shape
    video_src.reset()

    # Set up video writer
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    video_out_filepath = output_path + name + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_out_filepath, fourcc, 25, (img_width, img_height))
    print('Writing video at {}.'.format(video_out_filepath))

    bbx_idx = 0
    tbar = tqdm.tqdm(range(max_frame))
    for frame_idx in tbar:

        ret, image = video_src.get_frame()
        if not ret:
            break

        if frame_idx in frame_dict.keys():
            bounding_boxes = frame_dict[frame_idx]

            for bbx in bounding_boxes:
                label = pred_labels[bbx_idx]
                annotation_str = ['Subject: {}'.format(label)]

                draw_bounding_box_on_image_array(
                    image,
                    bbx[1],
                    bbx[0],
                    bbx[3],
                    bbx[2],
                    color=color[label],
                    thickness=4,
                    display_str_list=annotation_str,
                    use_normalized_coordinates=False)

                bbx_idx += 1

        cv2.putText(image, "Frame {}".format(frame_idx), (10, 20),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
        out.write(image)
    out.release()

def annotate_video(movie_file_path: str,
                   dataset_path: str,
                   output_path: str,
                   model: nn.Module,
                   device,
                   max_frame: int = 100000,
                   tracker_max_age: int = 10,
                   plotter: utils.plotter_utils.VisdomPlotter = None,
                   name: str = '',
                   compute_track_mean: bool = False):

    filename = os.path.join(dataset_path, 'bbx.txt')
    print('Getting annotations from {}'.format(filename))
    bbx_list = utils.read_file_to_list(filename)

    if bbx_list:
        bounding_boxes_list = bbx_list
    else:
        bounding_boxes_list = get_bounding_boxes(movie_file_path,
                                                 max_frame=max_frame,
                                                 tracker_max_age=tracker_max_age)

    print('Extracting ROI of the video.')
    cropped_image_list = get_cropped_images(movie_file_path,
                                            bounding_boxes_list,
                                            max_frame=max_frame)

    track_dict = get_track_dict(bounding_boxes_list)
    frame_dict = get_frame_dict(bounding_boxes_list)
    bbx_dict = get_bbx_dict(bounding_boxes_list)

    # Data transform
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = NumpyDataset(cropped_image_list,
                           transform=data_transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             num_workers=2,
                                             batch_size=100)

    print('Extracting features.')
    model = model.to(device)
    features = ml_utils.extract_features(dataloader,
                                         model,
                                         device)
    cluster_techniques_list = ['kmeans', 'spectral', 'hac']

    tsne_features, tsne_chosen_samples = projection_utils.tsne_projection(features)
    pca_features, pca_chosen_samples = projection_utils.pca_projection(features)

    # Frame level clustering
    print('Performing frame level clustering.')
    for cluster_method in cluster_techniques_list:
        cluster_name = '{}_frame_level_{}'.format(name, cluster_method)
        predictions, data_dict = clustering.cluster_techniques(features,
                                                     cluster_method,
                                                     max_clusters=10)

        write_video(movie_file_path,
                    output_path,
                    predictions,
                    frame_dict,
                    name=cluster_name,
                    max_frame=max_frame)

        plotter.scatter_plot(cluster_name + '_tsne',
                             tsne_features,
                             predictions[tsne_chosen_samples])
        plotter.scatter_plot(cluster_name + '_pca',
                             pca_features,
                             predictions[pca_chosen_samples])

    # Add ground truth if it exist
    gt_file_path = os.path.join(dataset_path, 'bbx_gt.txt')
    if os.path.isfile(gt_file_path):
        print('Creating ground truth video and plots.')
        bbx_to_gt_list = utils.read_file_to_list(gt_file_path)
        bbx_to_gt_dict = utils.list_to_dict(bbx_to_gt_list)

        groundtruth = []
        gt_to_idx_dict = {}
        bbx_count = 0
        for bbx in bounding_boxes_list:
            bbx_idx = bbx[2]
            gt = bbx_to_gt_dict[bbx_idx]
            if gt not in gt_to_idx_dict.keys():
                gt_to_idx_dict[gt] = bbx_count
                bbx_count += 1
            label = gt_to_idx_dict[gt]
            groundtruth.append(label)
        groundtruth = np.array(groundtruth)

        gt_name = '{}_gt'.format(name)
        write_video(movie_file_path,
                    output_path,
                    groundtruth,
                    frame_dict,
                    name=gt_name,
                    max_frame=max_frame)

        plotter.scatter_plot(gt_name + '_tsne',
                             tsne_features,
                             groundtruth[tsne_chosen_samples])
        plotter.scatter_plot(gt_name + '_pca',
                             pca_features,
                             groundtruth[pca_chosen_samples])

    # Track level clustering
    if compute_track_mean:
        print('Performing track level clustering.')

        mean_features = []
        track_to_idx_dict = {}
        for idx, track_idx in enumerate(track_dict.keys()):
            feature_track = features[track_dict[track_idx]]
            mean_features.append(np.mean(feature_track, axis=0))
            track_to_idx_dict[track_idx] = idx
        mean_features = np.asarray(mean_features)

        for cluster_method in cluster_techniques_list:
            cluster_name = '{}_track_level_{}'.format(name, cluster_method)
            mean_predictions, data_dict = clustering.cluster_techniques(mean_features,
                                                                        cluster_method,
                                                                        max_clusters=10)
            predictions = []
            for bbx_idx in bbx_dict.keys():
                track_idx = track_to_idx_dict[bbx_dict[bbx_idx][0]]
                predictions.append(mean_predictions[track_idx])
            predictions = np.array(predictions)

            write_video(movie_file_path,
                        output_path,
                        predictions,
                        frame_dict,
                        name=cluster_name,
                        max_frame=max_frame)

            plotter.scatter_plot(cluster_name + '_tsne',
                                 tsne_features,
                                 predictions[tsne_chosen_samples])
            plotter.scatter_plot(cluster_name + '_pca',
                                 pca_features,
                                 predictions[pca_chosen_samples])
