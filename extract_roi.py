
import argparse
import sys
import os
import json
import utils

from utils import roi_utils


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

    print('ROI extraction.')
    print('CONFIGURATION:\t{}'.format(args.config))
    with open(args.config) as json_config_file:
        config = utils.AttrDict(json.load(json_config_file))

    if not os.path.exists(config.output.output_dir):
        os.mkdir(config.output.output_dir)

    if args.mode == 0:
        roi_utils.extract_roi(
            config.dataset.movie.movie_path,
            config.dataset.movie.dataset_path,
            max_frame=config.dataset.movie.num_frame,
            tracker_max_age=config.hyperparameters.tracker_max_age)
    elif args.mode == 1:
        roi_utils.extract_roi_from_matlab_annotations(
            config.dataset.bbt.movie_path,
            config.dataset.bbt.annotation_path,
            config.dataset.bbt.dataset_path,
            config.dataset.bbt.num_frame)
    else:
        raise Exception('Mode {} is invalid.'.format(args.mode))
