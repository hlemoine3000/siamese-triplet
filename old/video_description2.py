
import argparse
import cv2
import os
import sys
import json
import tqdm
from PIL import Image
from datetime import datetime
import numpy as np

import torch
from torchvision import transforms
from torchvision.transforms.transforms import F
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.cluster import v_measure_score

import utils
from utils.fast_dt import FAST_DT
from utils.video import Video_Reader
from utils.visualization import draw_bounding_box_on_image_array
from ml_utils import ml_utils, clustering
import models
from dataset_utils.dataset import NumpyDataset

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


def annotate_video(video_src: Video_Reader,
                   frame_track_dict: dict,
                   labels,
                   video_out_path):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_out_path, fourcc, 25, (img_width, img_height))

    video_src.reset()
    bbx_idx = 0
    tbar = tqdm.tqdm(range(num_frame))
    for frame_idx in tbar:

        ret, image = video_src.get_frame()
        if not ret:
            break

        if frame_idx in frame_track_dict.keys():
            bounding_boxes = frame_track_dict[frame_idx]

            for bbx in bounding_boxes:
                label = labels[bbx_idx]

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
    print('')

    print('Saving video at ' + video_out_path)
    out.release()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/video_description.json')

    return parser.parse_args(argv)


if __name__ == '__main__':

    # ffmpeg - i bbtS01E01.mkv - vf scale = 1024:576 - r 25 - codec: a copy outputbbt1.mkv

    args = parse_arguments(sys.argv[1:])

    print('Feature extractor evaluation.')
    print('CONFIGURATION:\t{}'.format(args.config))
    with open(args.config) as json_config_file:
        config = utils.AttrDict(json.load(json_config_file))

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load model
    print('Loading model from checkpoint {}'.format(config.model.checkpoint_path))
    checkpoint = torch.load(config.model.checkpoint_path)
    embedding_size = checkpoint['embedding_size']

    model = models.load_model(config.model.model_arch,
                              device,
                              embedding_size=embedding_size)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create video source instance
    print('Initializing video capture at {}'.format(config.dataset.bbt.movie_path))
    video_src = Video_Reader(config.dataset.bbt.movie_path)

    if not os.path.exists(config.output.video_dir):
        os.mkdir(config.output.video_dir)

    _, image = video_src.get_frame()

    img_height, img_width, img_channel = image.shape

    # time metrics
    cycle_time = 1.0

    cropped_image_list = []
    track_dict = {}
    frame_dict = {}
    num_frame = config.dataset.bbt.num_frame

    my_fastdt = FAST_DT(tracker_max_age=config.hyperparameters.tracker_max_age)

    print('Generate bounding boxes.')

    # Generate bounding boxes for each frame of the movie
    video_src.reset()
    frame_idx = 0
    tbar = tqdm.tqdm(range(num_frame))
    for j in tbar:
    # for j in range(num_frame):

        # print('Frame {}'.format(frame_idx))
        ret, image = video_src.get_frame()
        if not ret:
            break

        bounding_boxes = my_fastdt.predict(image)

        frame_track_list = []
        for bbx in bounding_boxes:
            # Add image index toactual track
            track_id = bbx[4]
            sample_data = np.concatenate((bbx[0:4], [frame_idx]))
            if track_id not in track_dict.keys():
                # Init Image Indexes list for the new track
                track_dict[track_id] = [sample_data]
            else:
                track_dict[track_id].append(sample_data)

        frame_idx += 1

    print('{} tracklets.'.format(len(track_dict.keys())))

    frame_track_dict = {}
    track_dict_keys = list(track_dict.keys())
    for track_id in track_dict_keys:
        # Remove the last samples of each track as they are residual samples from the tracker
        if len(track_dict[track_id]) < config.hyperparameters.tracker_max_age:
            track_dict.pop(track_id)
        else:
            del track_dict[track_id][-config.hyperparameters.tracker_max_age]
            # track_dict[track_id] = track_dict[track_id][0:config.hyperparameters.tracker_max_age]
            # Distribute track samples into a frame list
            for sample_data in track_dict[track_id]:
                frame_idx = sample_data[4]
                data = np.concatenate((sample_data[0:4], [track_id]))
                if frame_idx not in frame_track_dict.keys():
                    frame_track_dict[frame_idx] = [data]
                else:
                    frame_track_dict[frame_idx].append(data)

    print('Extracting face patches.')

    video_src.reset()
    frame_idx = 0
    tbar = tqdm.tqdm(range(num_frame))
    for j in tbar:

        ret, image = video_src.get_frame()
        if not ret:
            break

        if frame_idx in frame_track_dict.keys():
            bounding_boxes = frame_track_dict[frame_idx]

            for bbx in bounding_boxes:

                cropped_image = image[bbx[1]: bbx[3], bbx[0]: bbx[2], :]
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                cropped_image = Image.fromarray(cropped_image)
                cropped_image = utils.make_square(cropped_image)
                cropped_image = F.resize(cropped_image, size=160, interpolation=1)
                cropped_image_list.append(cropped_image)

        frame_idx += 1
    print('')
    print('{} tracks.'.format(len(list(track_dict.keys()))))

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

    # Clustering
    print('Performing clustering.')
    pred_labels, nb_cluster = clustering.cluster_techniques(features,
                                                            cluster_methods='kmeans')
    cluster_classes = np.unique(pred_labels)
    print('{} clusters.'.format(nb_cluster))

    # # Frame level clustering
    # print('Performing frame level clustering.')
    #
    # pred_labels = my_clustering.cluster(features)
    # cluster_classes = np.unique(pred_labels)
    #
    # # Track level clustering
    # print('Performing track level clustering.')
    #
    # mean_feature_tracklist = []
    # for track_idx in track_dict.keys():
    #     feature_track = features[track_dict[track_idx]]
    #     mean_feature_tracklist.append(np.mean(feature_track, axis=0))
    # mean_feature_tracklist = np.asarray(mean_feature_tracklist)
    #
    # pred_tracklabels = my_clustering.cluster(mean_feature_tracklist)
    # cluster_classes = np.unique(pred_tracklabels)
    #
    # tsne_vis = TSNE_Visualizer()
    # tsne_vis.train(mean_feature_tracklist)
    # tsne_vis.plot(pred_tracklabels, env_name='BBT_vis', name='Emb_visPred')

    print('Describing movie.')

    filename = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    video_out_filepath = config.output.video_dir + filename + '.avi'
    annotate_video(video_src,
                   frame_track_dict,
                   pred_labels,
                   video_out_filepath)

    print('Process completed.')
