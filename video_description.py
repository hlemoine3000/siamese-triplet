
import argparse
import os
import sys
import json

import torch

import utils
from utils import roi_utils
from utils import vd_utils
import models
import train

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/video_description.json')

    return parser.parse_args(argv)


if __name__ == '__main__':

    # ffmpeg - i bbtS01E01.mkv - vf scale = 1024:576 - r 25 - codec: a copy outputbbt1.mkv

    args = parse_arguments(sys.argv[1:])

    with open(args.config) as json_config_file:
        config = utils.AttrDict(json.load(json_config_file))
    print('Complete video description process.')
    print('CONFIGURATION:\t{}\n'.format(args.config))

    #####################################
    if os.path.isfile(os.path.join(config.dataset.movie.dataset_path, 'cooccurring_tracks.txt')):
        print('')
        print('Skipping ROI extraction. Dataset already generated at {}'.format(config.dataset.bbt.dataset_path))
    else:
        print('')
        print('ROI extraction.')
        if not os.path.exists(config.output.output_dir):
            os.mkdir(config.output.output_dir)

        roi_utils.extract_roi(
            config.dataset.movie.movie_path,
            config.dataset.movie.dataset_path,
            max_frame=config.dataset.movie.num_frame,
            tracker_max_age=config.hyperparameters.tracker_max_age)

    #####################################
    print('')
    print('Describe movie with base FE.')

    # Load model
    print('Loading model from checkpoint {}'.format(config.model.checkpoint_path))
    checkpoint = torch.load(config.model.checkpoint_path)
    embedding_size = checkpoint['embedding_size']

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = models.load_model(config.model.model_arch,
                              device,
                              embedding_size=embedding_size)
    model.load_state_dict(checkpoint['model_state_dict'])

    filename = os.path.join(config.dataset.movie.dataset_path, 'bbx.txt')
    bbx_list = utils.read_file_to_list(filename)

    plotter = utils.VisdomPlotter(config.visdom.server, env_name='video_annotation', port=config.visdom.port)

    vd_utils.annotate_video(config.dataset.movie.movie_path,
                            config.output.video_dir,
                            model,
                            device,
                            max_frame=config.dataset.movie.num_frame,
                            bbx_list=bbx_list,
                            tracker_max_age=config.hyperparameters.tracker_max_age,
                            plotter=plotter,
                            name='base')

    #####################################
    # print('')
    # print('Finetune model')
    # model = train.main(args)

    #####################################
    # print('')
    # print('Describe movie with finetuned FE.')
    #
    # vd_utils.annotate_video(config.dataset.movie.movie_path,
    #                         config.output.video_dir,
    #                         model,
    #                         device,
    #                         max_frame=config.dataset.movie.num_frame,
    #                         bbx_list=bbx_list,
    #                         tracker_max_age=config.hyperparameters.tracker_max_age,
    #                         plotter=plotter,
    #                         name='finetune')
