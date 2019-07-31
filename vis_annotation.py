
import argparse
import configparser
import cv2
import os
import time
import sys
import signal
from datetime import datetime

from utils.video import Video_Reader
from utils.visualization import draw_bounding_box_on_image_array
from dataset_utils.bbt import Read_Annotation

images_path = 'images/'
database_output_path = 'database/'


def signal_handler(sig, frame):
    if out:
        out.release()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':

    # ffmpeg - i bbtS01E01.mkv - vf scale = 1024:576 - r 25 - codec: a copy outputbbt1.mkv

    annotation_path = '/export/livia/data/lemoineh/CVPR2013_PersonID_data/bbt_s01e01_facetracks.mat'
    movie_path = '/export/livia/data/lemoineh/BBT/bbts01e01.mkv'
    dataset_path = '/export/livia/data/lemoineh/BBT/ep01/'


    # Create video source instance
    print('Initializing video capture.')
    video_src = Video_Reader(movie_path)

    bbx = None

    video_out_dir = 'output/'
    if not os.path.exists(video_out_dir):
        os.mkdir(video_out_dir)

    _, image = video_src.get_frame()

    img_height, img_width, img_channel = image.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filename = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    video_out_filepath = video_out_dir + filename + '.avi'
    out = cv2.VideoWriter(video_out_filepath, fourcc, 25, (img_width, img_height))

    groundtruth_dict = {}

    Annotation_list = Read_Annotation(annotation_path, (img_width, img_height))

    # time metrics
    cycle_time = 1.0

    print('\nApplication running.\n')
    print('Describing video sequence.')

    video_src.reset()
    ret, image = video_src.get_frame()
    frame_idx = 0
    while ret and (frame_idx < 1000):

        annotation_idx = frame_idx - 23
        if annotation_idx < 0:
            frame_annotations = []
        else:
            frame_annotations = Annotation_list[annotation_idx]

        for annotation in frame_annotations:

            cropped_image = image[annotation[1]: annotation[3], annotation[0]: annotation[2],:]
            track_name = 'track_{:06d}'.format(annotation[6])
            image_name = 'frame_{:06d}.png'.format(annotation[5])
            image_dir = os.path.join(dataset_path, track_name)
            image_path = os.path.join(dataset_path, track_name, image_name)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            cv2.imwrite(image_path, cropped_image)

            if track_name not in groundtruth_dict.keys():
                groundtruth_dict[track_name] = annotation[4]

            draw_bounding_box_on_image_array(
                image,
                annotation[1],
                annotation[0],
                annotation[3],
                annotation[2],
                color='Pink',  # This is not pink !?!? You lying to me?!
                thickness=4,
                display_str_list=[annotation[4]],
                use_normalized_coordinates=False)

        cv2.putText(image, "Frame {}".format(frame_idx), (10, 20),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)

        out.write(image)
        sys.stdout.write("\r" + 'Frame {} processed.'.format(frame_idx))
        sys.stdout.flush()

        ret, image = video_src.get_frame()
        frame_idx += 1

    groundtruth_filepath = os.path.join(dataset_path, 'groundtruth.txt')
    groundtruth_file = open(groundtruth_filepath, "w")
    for key in groundtruth_dict.keys():
        groundtruth_file.writelines('{}\t{}\n'.format(key, groundtruth_dict[key]))
    groundtruth_file.close()

    print('')
    print('Saving video at ' + video_out_filepath)
    out.release()

    print('Process completed.')
