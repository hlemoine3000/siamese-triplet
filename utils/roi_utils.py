
import cv2
import os
import tqdm
from PIL import Image

from torchvision.transforms.transforms import F

import utils
from utils.fast_dt import FAST_DT
from utils.video import Video_Reader
from dataset_utils import bbt


def extract_roi_from_matlab_annotations(movie_path: str,
                                        annotation_path: str,
                                        output_path: str,
                                        max_frame: int=100000):

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Create video source instance
    print('Initializing video capture at {}'.format(movie_path))
    video_src = Video_Reader(movie_path)
    _, image = video_src.get_frame()
    video_src.reset()

    img_height, img_width, img_channel = image.shape

    print('Reading annotation at {}'.format(annotation_path))
    Annotation_list = bbt.Read_Annotation(annotation_path, (img_width, img_height))

    cooccurring_tracks = []
    bounding_boxes_list = []
    bbx_to_gt_list = []
    track_to_gt_list = []

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

        track_list = []
        for bbx in bounding_boxes:

            cropped_image = image[bbx[1]: bbx[3], bbx[0]: bbx[2], :]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = Image.fromarray(cropped_image)
            cropped_image = utils.make_square(cropped_image)
            cropped_image = cropped_image.resize((160, 160), resample=Image.LANCZOS)

            track_id = bbx[6]
            gt_label = bbx[4]
            bounding_boxes_list.append([frame_idx, track_id, bbx_idx, bbx[0], bbx[1], bbx[2], bbx[3]])
            bbx_to_gt_list.append([bbx_idx, gt_label])
            track_to_gt_list.append([track_id, gt_label])

            # Save image
            dir_name = '{:04d}'.format(track_id)
            image_name = '{:06d}.png'.format(bbx_idx)
            save_path = os.path.join(output_path, dir_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_file_path = os.path.join(save_path, image_name)
            cropped_image.save(save_file_path)

            track_list.append(track_id)
            bbx_idx += 1

        # Note co-occurring tracks
        if len(track_list) > 1:
            track_list = sorted(track_list)
            if track_list not in cooccurring_tracks:
                cooccurring_tracks.append(track_list)

        frame_idx += 1

    # Save co-occurring tracksset
    utils.write_list_to_file(os.path.join(output_path, "cooccurring_tracks.txt"),
                             cooccurring_tracks)
    # Save bbx
    utils.write_list_to_file(os.path.join(output_path, "bbx.txt"),
                             bounding_boxes_list)

    # Save ground truth
    utils.write_list_to_file(os.path.join(output_path, "bbx_gt.txt"),
                             bbx_to_gt_list)
    utils.write_list_to_file(os.path.join(output_path, "track_gt.txt"),
                             track_to_gt_list)

    print('{} co-occurring tracks.'.format(len(cooccurring_tracks)))


def extract_roi(movie_path: str,
                output_path: str,
                max_frame: int=100000,
                tracker_max_age: int=10):

    # Create video source instance
    print('Initializing video capture at {}'.format(movie_path))
    video_src = Video_Reader(movie_path)
    _, image = video_src.get_frame()
    video_src.reset()

    my_fastdt = FAST_DT("cpu", tracker_max_age=tracker_max_age)

    print('Extracting face patches.')

    image_dict = {}
    bbx_dict = {}
    cooccurring_tracks = []
    bbx_idx = 0
    tbar = tqdm.tqdm(range(max_frame))
    for frame_idx in tbar:

        ret, image = video_src.get_frame()
        if not ret:
            break

        bounding_boxes = my_fastdt.predict(image)

        for bbx in bounding_boxes:

            cropped_image = image[bbx[1]: bbx[3], bbx[0]: bbx[2], :]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = Image.fromarray(cropped_image)
            cropped_image = utils.make_square(cropped_image)
            cropped_image = F.resize(cropped_image, size=160, interpolation=1)

            track_id = bbx[4]
            # bounding_boxes_list.append([frame_idx, track_id, bbx_idx, bbx[0], bbx[1], bbx[2], bbx[3]])

            if track_id not in image_dict.keys():
                image_dict[track_id] = [(cropped_image, bbx_idx, frame_idx)]
                bbx_dict[track_id] = [[frame_idx, track_id, bbx_idx, bbx[0], bbx[1], bbx[2], bbx[3]]]
            else:
                image_dict[track_id].append((cropped_image, bbx_idx, frame_idx))
                bbx_dict[track_id].append([frame_idx, track_id, bbx_idx, bbx[0], bbx[1], bbx[2], bbx[3]])

            bbx_idx += 1

    # Remove the last samples of each track as they are residual samples from the tracker max age
    print('Removing residual samples.')
    track_id_list = list(image_dict.keys())
    for track_id in track_id_list:
        if len(image_dict[track_id])+1 < tracker_max_age:
            image_dict.pop(track_id)
            bbx_dict.pop(track_id)
        else:
            image_dict[track_id] = image_dict[track_id][1:-tracker_max_age]
            bbx_dict[track_id] = bbx_dict[track_id][1:-tracker_max_age]

    # Create the bounding_box_list
    bounding_boxes_list = []
    for track_id in bbx_dict.keys():
        for bbx in bbx_dict[track_id]:
            bounding_boxes_list.append(bbx)

    # Convert the track classed dictionary to a frame classed dictionary
    print('Creating dataset.')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    frame_to_track_dict = {}
    tbar2 = tqdm.tqdm(image_dict.keys())
    for track_id in tbar2:
        for cropped_image, bbx_idx, frame_idx in image_dict[track_id]:
            if frame_idx not in frame_to_track_dict.keys():
                frame_to_track_dict[frame_idx] = [track_id]
            else:
                frame_to_track_dict[frame_idx].append(track_id)

            # Save image
            dir_name = '{:04d}'.format(track_id)
            image_name = '{:06d}.png'.format(bbx_idx)
            save_path = os.path.join(output_path, dir_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_file_path = os.path.join(save_path, image_name)
            cropped_image.save(save_file_path)

    # Find co-occurring tracks
    print('Forming co-occurring tracks file.')
    for frame_idx in frame_to_track_dict.keys():
        track_list = []
        for track_id in frame_to_track_dict[frame_idx]:
            track_list.append(track_id)
        # Note co-occurring tracks
        if len(track_list) > 1:
            track_list = sorted(track_list)
            if track_list not in cooccurring_tracks:
                cooccurring_tracks.append(track_list)

        # Save co-occurring tracksset
        utils.write_list_to_file(os.path.join(output_path, "cooccurring_tracks.txt"),
                                 cooccurring_tracks)
        # Save bbx
        utils.write_list_to_file(os.path.join(output_path, "bbx.txt"),
                                 bounding_boxes_list)

    print('{} co-occurring tracks.'.format(len(cooccurring_tracks)))
