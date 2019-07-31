
import argparse
import cv2
import os
import sys
import signal
import json
import tqdm
import pickle
from datetime import datetime
import numpy as np

import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.cluster import v_measure_score

import utils
from utils.video import Video_Reader
from utils.visualization import draw_bounding_box_on_image_array
from ml_utils import ml_utils, clustering
import models
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

    # Create video source instance
    print('Initializing video capture at {}'.format(config.dataset.bbt.movie_path))
    video_src = Video_Reader(config.dataset.bbt.movie_path)

    _, image = video_src.get_frame()

    img_height, img_width, img_channel = image.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filename = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    if not os.path.exists(config.output.video_dir):
        os.mkdir(config.output.video_dir)
    video_out_filepath = config.output.video_dir + filename + '.avi'
    out = cv2.VideoWriter(video_out_filepath, fourcc, 25, (img_width, img_height))

    roi_file = os.path.join(config.output.output_dir, 'roi.pkl')
    file2 = open(roi_file, 'rb')
    data = pickle.load(file2)
    file2.close()

    # data_dict = {'frame_annotations': frame_dict,  # [frame index]: BBX + track id
    #              'track_to_bbxidx': track_dict,  # [track id]: cropped image indexes
    #              'cropped_images': cropped_image_list,  # Cropped image list
    #              'track_id': track_id,  # Track id associated with cropped image index
    #              'cooccurring_tracks': cooccurring_tracks}  # List of list of co-occurring track id

    frame_dict = data['frame_annotations']
    track_to_bbxidx = data['track_to_bbxidx']
    cropped_image_list = data['cropped_images']

    num_gt_classes = 7

    # Data transform
    data_transform = transforms.Compose([
        transforms.Resize((160, 160), interpolation=1),
        transforms.ToTensor()
    ])

    dataset = NumpyDataset(cropped_image_list,
                           transform=data_transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             num_workers=2,
                                             batch_size=100,
                                             pin_memory=True)

    print('Extracting features.')
    model = model.to(device)
    features = ml_utils.extract_features(dataloader,
                                         model,
                                         device)

    # Clustering
    nb_cluster = num_gt_classes
    my_clustering = clustering.hac_cluster(nb_cluster)

    # Frame level clustering
    print('Performing frame level clustering.')
    pred_labels = my_clustering.cluster(features)
    cluster_classes = np.unique(pred_labels)

    # Track level clustering
    print('Performing track level clustering.')
    mean_feature_tracklist = []
    for track_idx in track_to_bbxidx.keys():
        feature_track = features[track_to_bbxidx[track_idx]]
        mean_feature_tracklist.append(np.mean(feature_track, axis=0))
    mean_feature_tracklist = np.asarray(mean_feature_tracklist)

    pred_tracklabels = my_clustering.cluster(mean_feature_tracklist)
    cluster_classes = np.unique(pred_tracklabels)

    # tsne_vis = TSNE_Visualizer()
    # tsne_vis.train(mean_feature_tracklist)
    # tsne_vis.plot(pred_tracklabels, env_name='BBT_vis', name='Emb_visPred')

    print('Describing video sequence.')

    video_src.reset()
    frame_idx = 0
    bbx_idx = 0

    tbar = tqdm.tqdm(range(config.dataset.bbt.num_frame))
    for j in tbar:

        ret, image = video_src.get_frame()
        if not ret:
            break

        bounding_boxes_list = frame_dict[frame_idx]

        for bbx in bounding_boxes_list:

            frame_label = pred_labels[bbx_idx]
            track_label = pred_tracklabels[bbx[4]]

            annotation_str = ['Frame: {}'.format(frame_label),
                              'Track: {}'.format(track_label)]

            draw_bounding_box_on_image_array(
                image,
                bbx[1],
                bbx[0],
                bbx[3],
                bbx[2],
                color='#66ff66',
                thickness=4,
                display_str_list=annotation_str,
                use_normalized_coordinates=False)

            bbx_idx += 1

        cv2.putText(image, "Frame {}".format(frame_idx), (10, 20),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
        out.write(image)

        frame_idx += 1
    print('')

    print('Saving video at ' + video_out_filepath)
    out.release()

    print('Process completed.')
