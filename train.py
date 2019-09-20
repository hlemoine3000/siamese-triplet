import sys
import os
import argparse
from datetime import datetime
from shutil import copyfile
import ntpath
import json

import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms

from experiments import tripletloss_exp
from ml_utils import trainer, losses, miners
import evaluation
import models
import utils


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/train_config.json')

    return parser.parse_args(argv)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def main(args):

    print('Feature extractor training.')
    print('CONFIGURATION:\t{}'.format(args.config))
    with open(args.config) as json_config_file:
        config = utils.AttrDict(json.load(json_config_file))

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # device = torch.device("cpu")

    # Get dataloaders
    online_train_loader, test_container = tripletloss_exp.Get_TrainDataloaders(config.experiment, config)

    # Set up training model
    print('Building training model')
    if config.model.checkpoint:
        checkpoint_path = config.model.checkpoint_path
    else:
        checkpoint_path = None
    model = models.load_model(config.model.model_arch,
                              device,
                              checkpoint_path=checkpoint_path,
                              embedding_size=config.model.embedding_size,
                              imgnet_pretrained=config.model.pretrained_imagenet)

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    optimizer = optim.SGD(model.parameters(), lr=config.hyperparameters.learning_rate, momentum=0.9, nesterov=True, weight_decay=2e-4)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, eps=1.0, weight_decay=weight_decay, momentum=0.9, centered=False)

    # scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    scheduler = lr_scheduler.ExponentialLR(optimizer, config.hyperparameters.learning_rate_decay_factor)

    plotter = utils.VisdomPlotter(env_name=config.visdom.environment_name, port=config.visdom.port)

    miner = miners.FunctionSemihardTripletSelector(config.hyperparameters.margin, plotter)

    loss = nn.TripletMarginLoss(config.hyperparameters.margin, swap=config.hyperparameters.triplet_swap)

    my_trainer = trainer.Triplet_Trainer(model,
                                         miner,
                                         loss,
                                         optimizer,
                                         scheduler,
                                         device,
                                         plotter,
                                         config.hyperparameters.margin,
                                         config.model.embedding_size,
                                         evaluation.pair_evaluate,
                                         batch_size=config.hyperparameters.batch_size)

    # Set up output directory
    # subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    subdir = config.visdom.environment_name
    model_dir = os.path.join(os.path.expanduser(config.output.output_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    # else:
    #     raise Exception('Environment name {} already taken.'.format(subdir))
    config_filename = path_leaf(args.config)
    copyfile(args.config, os.path.join(model_dir, config_filename))

    # Loop over epochs
    epoch = 0
    print('Training Launched.')
    while epoch < config.hyperparameters.n_epochs:

        # Validation
        for test_name, test_loader, eval_function in test_container:
            print('\nEvaluation on {}'.format(test_name))
            eval_function(test_loader,
                          model,
                          device,
                          test_name,
                          plotter=plotter,
                          epoch=epoch,
                          distance_metric=0,
                          val_far=config.hyperparameters.val_far)

        # Training
        print('\nTrain Epoch {}'.format(epoch))
        my_trainer.Train_Epoch(online_train_loader, epoch)

        # Save model
        if not (epoch + 1) % config.output.save_interval:
            model_file_path = os.path.join(model_dir, 'model_{}.pth'.format(epoch))
            print('\nSave model at {}'.format(model_file_path))

            torch.save({'epoch': epoch,
                        'model_state_dict': utils.state_dict_to_cpu(model.state_dict()),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'embedding_size': config.model.embedding_size,
                        }, model_file_path)

        epoch += 1

    # Final save.
    model_file_path = os.path.join(model_dir, 'model_{}.pth'.format(epoch))
    print('\nSave model at {}'.format(model_file_path))
    torch.save({'epoch': epoch,
                'model_state_dict': utils.state_dict_to_cpu(model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'embedding_size': config.model.embedding_size,
                }, model_file_path)
    print('Finish.')

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
