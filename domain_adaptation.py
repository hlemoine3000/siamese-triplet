import sys
import os
import argparse
from datetime import datetime
from shutil import copyfile
import ntpath
import json

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from experiments import da_exp
from ml_utils import trainer, losses, miners, clustering
from dataset_utils import dataloaders
import models
import utils


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/da_config.json')

    return parser.parse_args(argv)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def generate_experiment_name(config):

    if config.debug:
        experiment_name = 'debug'
    else:
        experiment_name = 'da_{}_to_{}_L'.format(config.source_dataset, config.target_dataset)
        if config.hyperparameters.lamda[0] > 0.0:
            experiment_name += '1'
        if config.hyperparameters.lamda[1] > 0.0:
            experiment_name += '2'

        experiment_name += '_m{}'.format(config.hyperparameters.margin)

        if config.miner == 'supervised_dualtriplet':
            experiment_name += '_supervised'

        if os.path.isdir(os.path.join(os.path.expanduser(config.output.output_dir), experiment_name)):
            dir_count = 1
            experiment_name += '_1'
            while os.path.isdir(os.path.join(os.path.expanduser(config.output.output_dir), experiment_name)):
                dir_count += 1
                experiment_name = experiment_name[:-2] + '_{}'.format(dir_count)

    return experiment_name


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

    config_filename = path_leaf(args.config)
    copyfile(args.config, os.path.join(model_dir, config_filename))

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    source_loader = dataloaders.get_traindataloaders(config.source_dataset,
                                                    config)
    target_loader = dataloaders.get_traindataloaders(config.target_dataset,
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

    scheduler = lr_scheduler.ExponentialLR(optimizer, config.hyperparameters.learning_rate_decay_factor)

    model = model.to(device)

    plotter = utils.VisdomPlotter(config.visdom.server ,env_name=experiment_name, port=config.visdom.port)

    print('Fitting source dataset.')
    gmixture = clustering.distance_supervised_gaussian_mixture(source_loader,
                                                               model,
                                                               device,
                                                               _plotter=plotter,
                                                               name='Source Gaussians')

    print('Fitting target dataset.')
    clustering.update_gaussian_mixture(gmixture,
                                       target_loader,
                                       model,
                                       device,
                                       _plotter=plotter,
                                       name='Target Gaussians')

    print('DualTriplet loss training mode.')
    miner = miners.get_miner(config.miner,
                             config.hyperparameters.margin,
                             config.hyperparameters.people_per_batch,
                             plotter,
                             deadzone_ratio=config.hyperparameters.deadzone_ratio)
    miner.gmixture = gmixture

    loss = losses.DualtripletLoss(config.hyperparameters.margin,
                                  config.hyperparameters.lamda,
                                  plotter)

    model_trainer = trainer.Dualtriplet_Trainer(model,
                                                miner,
                                                loss,
                                                optimizer,
                                                scheduler,
                                                device,
                                                plotter,
                                                config.hyperparameters.margin,
                                                config.model.embedding_size,
                                                batch_size=config.hyperparameters.batch_size)

    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

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
        print('\nExperimentation {}'.format(config.experiment))
        print('Train Epoch {}'.format(epoch))
        model_trainer.Train_Epoch(source_loader, target_loader, epoch)

        # Save model
        # if not (epoch + 1) % config.output.save_interval:
        #
        #     model_file_path = os.path.join(model_dir, 'model_{}.pth'.format(epoch))
        #     print('\nSave model at {}'.format(model_file_path))
        #     torch.save({'epoch': epoch,
        #                 'model_state_dict': utils.state_dict_to_cpu(model.state_dict()),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'scheduler_state_dict': scheduler.state_dict(),
        #                 'embedding_size': config.model.embedding_size
        #                 }, model_file_path)

        epoch += 1

    model_file_path = os.path.join(model_dir, 'model_{}.pth'.format(epoch))
    print('\nSave model at {}'.format(model_file_path))
    torch.save({'epoch': epoch,
                'model_state_dict': utils.state_dict_to_cpu(model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'embedding_size': config.model.embedding_size
                }, model_file_path)
    print('Finish.')

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))