
import pickle
import argparse
import cv2
import sys
import os
import json
import tqdm
from PIL import Image

import torch
from torchvision.transforms.transforms import F

import utils
from utils.fast_dt import FAST_DT
from utils.video import Video_Reader
from dataset_utils import bbt


def Extract_LabeledROI(movie_path, annotation_path, max_frame):

    # Create video source instance
    print('Initializing video capture at {}'.format(movie_path))
    video_src = Video_Reader(movie_path)
    _, image = video_src.get_frame()
    video_src.reset()

    img_height, img_width, img_channel = image.shape

    print('Reading annotation at {}'.format(annotation_path))
    Annotation_list = bbt.Read_Annotation(annotation_path, (img_width, img_height))

    cooccurring_tracks = []
    track_id = []
    cropped_image_list = []
    track_dict = {}
    gt_tracklabels_list = []
    frame_dict = {}
    gt_labels_list = []

    classes_to_idx = {'sheldon': 0,
                      'leonard': 1,
                      'unknown': 2,
                      'penny': 3,
                      'raj': 4,
                      'howard': 5,
                      'kurt': 6}

    print('Extracting face patches.')

    frame_idx = 0
    bbx_idx = 0
    num_frame = min(len(Annotation_list), max_frame)
    tbar = tqdm.tqdm(range(num_frame))
    for j in tbar:

        ret, image = video_src.get_frame()
        if not ret:
            break

        bounding_boxes = Annotation_list[frame_idx]

        bbx_list = []
        track_list = []
        for bbx in bounding_boxes:

            cropped_image = image[bbx[1]: bbx[3], bbx[0]: bbx[2], :]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = Image.fromarray(cropped_image)
            cropped_image = utils.make_square(cropped_image)
            cropped_image = cropped_image.resize((160, 160), resample=Image.LANCZOS)
            cropped_image_list.append(cropped_image)
            track_id.append(bbx[6])

            # save_dir = '/export/livia/data/lemoineh/BBT/ep01_2/track{}_{}/'.format(bbx[6], bbx[4])
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            # save_file = os.path.join(save_dir, '{}.png'.format(bbx_idx))
            # cropped_image.save(save_file)

            # Add image index toactual track
            if bbx[6] not in track_dict.keys():
                # Init Image Indexes list for the new track
                gt_tracklabels_list.append(classes_to_idx[bbx[4]])
                track_dict[bbx[6]] = [bbx_idx]
            else:
                track_dict[bbx[6]].append(bbx_idx)

            gt_labels_list.append(classes_to_idx[bbx[4]])

            bbx_list.append(bbx)
            track_list.append(bbx[6])
            bbx_idx += 1

        # Note co-occurring tracks
        if len(bounding_boxes) > 1:
            track_list.sort()
            if track_list not in cooccurring_tracks:
                cooccurring_tracks.append(track_list)

        frame_dict[frame_idx] = bbx_list
        frame_idx += 1
    print('')
    print('{} tracks.'.format(len(list(track_dict.keys()))))

    # Remove duplicate co-occurring tracks
    print('Filtering co-occuring tracks.')
    elem_to_pop = []
    for k, cooccurring_track1 in enumerate(cooccurring_tracks[:-1]):
        for cooccurring_track2 in cooccurring_tracks[k+1:]:
            if all(elem in cooccurring_track2 for elem in cooccurring_track1):
                if k not in elem_to_pop:
                    elem_to_pop.append(k)
    elem_to_pop.sort(reverse=True)
    for elem_idx in elem_to_pop:
        # print('pop_idx: {}  list_length: {}'.format(elem_idx, len(cooccurring_tracks)))
        cooccurring_tracks.pop(elem_idx)

    print('{} co-occurring tracks.'.format(len(cooccurring_tracks)))

    data_dict = {'frame_annotations': frame_dict,  # [frame index]: BBX + track id
                 'track_to_bbxidx': track_dict,  # [track id]: cropped image indexes
                 'cropped_images': cropped_image_list,  # Cropped image list
                 'track_id': track_id,  # Track id associated with cropped image index
                 'cooccurring_tracks': cooccurring_tracks, # List of list of co-occurring track id
                 'gt_labels_list': gt_labels_list,
                 'gt_tracklabels_list': gt_tracklabels_list}

    return data_dict


def Extract_ROI(movie_path, num_frame):

    # CUDA for PyTorch
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    device = "cpu"

    # Create video source instance
    print('Initializing video capture at {}'.format(movie_path))
    video_src = Video_Reader(movie_path)
    _, image = video_src.get_frame()
    video_src.reset()

    cropped_image_list = []
    cooccurring_tracks = []
    track_id = []
    track_dict = {}
    frame_dict = {}

    my_fastdt = FAST_DT(device, tracker_max_age=10)

    print('Extracting face patches.')

    frame_idx = 0
    bbx_idx = 0
    tbar = tqdm.tqdm(range(num_frame))
    for j in tbar:

        ret, image = video_src.get_frame()
        if not ret:
            break

        bounding_boxes = my_fastdt.predict(image)

        bbx_list = []
        track_list = []
        for bbx in bounding_boxes:

            cropped_image = image[bbx[1]: bbx[3], bbx[0]: bbx[2], :]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = Image.fromarray(cropped_image)
            cropped_image = utils.make_square(cropped_image)
            cropped_image = F.resize(cropped_image, size=160, interpolation=1)
            cropped_image_list.append(cropped_image)
            track_id.append(bbx[4])

            # Add image index to actual track
            if bbx[4] not in track_dict.keys():
                # Init Image Indexes list for the new track
                track_dict[bbx[4]] = [bbx_idx]
            else:
                track_dict[bbx[4]].append(bbx_idx)

            bbx_list.append(bbx)
            track_list.append(bbx[4])
            bbx_idx += 1

        # Note co-occurring tracks
        if len(bounding_boxes) > 1:
            track_list.sort()
            if track_list not in cooccurring_tracks:
                cooccurring_tracks.append(track_list)

        frame_dict[frame_idx] = bbx_list
        frame_idx += 1
    print('')
    print('{} tracks.'.format(len(list(track_dict.keys()))))

    data_dict = {'frame_annotations': frame_dict,  # [frame index]: BBX + track id
                 'track_to_bbxidx': track_dict,  # [track id]: cropped image indexes
                 'cropped_images': cropped_image_list,  # Cropped image list
                 'track_id': track_id,  # Track id associated with cropped image index
                 'cooccurring_tracks': cooccurring_tracks}  # List of list of co-occurring track id

    return data_dict

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=int,
                        help='Path to the configuration file', default=0)
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

    if not os.path.exists(config.output.output_dir):
        os.mkdir(config.output.output_dir)

    if args.mode == 0:
        data_dict = Extract_ROI(config.dataset.bbt.movie_path,
                                config.dataset.bbt.num_frame)
        output_file = os.path.join(config.output.output_dir, 'roi.pkl')
    elif args.mode == 1:
        data_dict = Extract_LabeledROI(config.dataset.bbt.movie_path,
                                       config.dataset.bbt.annotation_path,
                                       config.dataset.bbt.num_frame)
        output_file = os.path.join(config.output.output_dir, 'labeled_roi.pkl')
    else:
        raise Exception('Mode {} is invalid.'.format(args.mode))

    file = open(output_file, 'wb+')
    pickle.dump(data_dict, file)
    file.close()
    print('Saved ROI at {}'.format(output_file))