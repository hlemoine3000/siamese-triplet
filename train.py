import sys
import os
import argparse
from shutil import copyfile
import json

import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler

from dataset_utils import dataloaders
from ml_utils import trainer, miners
import evaluation
import models
import utils


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/train_config.json')

    return parser.parse_args(argv)

def generate_experiment_name(config):

    if config.debug:
        experiment_name_test = 'debug'
    else:
        experiment_name = 'train_{}_m{}'.format(config.train_dataset, config.hyperparameters.margin)

        dir_count = 1
        experiment_name_test = experiment_name + '_{:02d}'.format(dir_count)
        while os.path.isdir(os.path.join(os.path.expanduser(config.output.output_dir), experiment_name_test)):
            dir_count += 1
            experiment_name_test = experiment_name + '_{:02d}'.format(dir_count)

    return experiment_name_test

def main(args):

    print('Feature extractor training.')
    print('CONFIGURATION:\t{}'.format(args.config))
    with open(args.config) as json_config_file:
        config = utils.AttrDict(json.load(json_config_file))

    # Set up output directory
    experiment_name = generate_experiment_name(config)
    model_dir = os.path.join(os.path.expanduser(config.output.output_dir), experiment_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Model saved at {}'.format(model_dir))

    config_filename = utils.path_leaf(args.config)
    copyfile(args.config, os.path.join(model_dir, config_filename))

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # device = torch.device("cpu")

    # Get dataloaders
    train_loader = dataloaders.get_traindataloaders(config.train_dataset,
                                                    config)
    evaluators_list = dataloaders.get_evaluators(config.evaluation_datasets,
                                                config)

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

    optimizer = optim.SGD(model.parameters(), lr=config.hyperparameters.learning_rate, momentum=0.9, nesterov=True, weight_decay=2e-4)

    # scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, config.hyperparameters.learning_rate_decay_factor)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.hyperparameters.n_epochs, eta_min=1e-6)

    plotter = utils.VisdomPlotter(config.visdom.server, env_name=experiment_name, port=config.visdom.port)

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

    # Loop over epochs
    epoch = 0
    print('Training Launched.')
    while epoch < config.hyperparameters.n_epochs:

        # Validation
        for evaluator in evaluators_list:
            print('\nEvaluation on {}'.format(evaluator.test_name))
            evaluator.evaluate(model,
                               device,
                               plotter=plotter,
                               epoch=epoch)

        # Training
        print('\nTrain Epoch {}'.format(epoch))
        my_trainer.Train_Epoch(train_loader, epoch)

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

    return model

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
