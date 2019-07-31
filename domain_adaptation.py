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
from ml_utils import trainer
from ml_utils import losses
from ml_utils import miners
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


def main(args):

    print('Feature extractor training.')
    print('CONFIGURATION:\t{}'.format(args.config))
    with open(args.config) as json_config_file:
        config = utils.AttrDict(json.load(json_config_file))

    # Set up output directory
    # subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    subdir = config.visdom.environment_name
    model_dir = os.path.join(os.path.expanduser(config.output.output_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
        print('Model saved at {}'.format(model_dir))
    # else:
    #     raise Exception('Environment name {} already taken.'.format(subdir))

    config_filename = path_leaf(args.config)
    copyfile(args.config, os.path.join(model_dir, config_filename))

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    nrof_folds = config.dataset.cross_validation.num_fold
    source_loader, target_loader, test_loaders_list = da_exp.Get_DADataloaders(config.experiment,
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

    plotter = utils.VisdomLinePlotter(env_name=config.visdom.environment_name, port=config.visdom.port)

    print('DualTriplet loss training mode.')
    miner = miners.get_miner(config.mode,
                             config.hyperparameters.margin,
                             config.hyperparameters.people_per_batch)

    loss = losses.DualtripletLoss(config.hyperparameters.margin,
                                  config.hyperparameters.lamda)

    model_trainer = trainer.get_trainer(config.mode,
                                        model,
                                        miner,
                                        loss,
                                        optimizer,
                                        scheduler,
                                        device,
                                        plotter,
                                        config.hyperparameters.margin,
                                        config.model.embedding_size,
                                        config.visdom.log_interval)

    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Loop over epochs
    epoch = 0
    print('Training Launched.')
    while epoch < config.hyperparameters.n_epochs:

        # Validation
        for test_name, test_loader in test_loaders_list:
            print('\nEvaluation on {}'.format(test_name))
            eval_data = model_trainer.Evaluate(test_loader,
                                         name=test_name,
                                         nrof_folds=nrof_folds,
                                         val_far=config.hyperparameters.val_far)

            plotter.plot('accuracy', 'epoch', test_name, 'Accuracy', epoch, eval_data['accuracy'])
            plotter.plot('auc', 'epoch', test_name, 'AUC', epoch, eval_data['auc'])

        # Training
        print('\nExperimentation {}'.format(config.experiment))
        print('Train Epoch {}'.format(epoch))
        train_data = model_trainer.Train_Epoch(source_loader, target_loader)

        if train_data is None:
            # Take last training results
            train_data = last_train_data
        else:
            last_train_data = train_data

        plotter.plot('learning rate', 'epoch', 'train', 'Learning Rate',
                     epoch, train_data['lr'])

        # Loss stats
        plotter.plot('loss', 'epoch', 'L12', 'Losses', epoch, train_data['L12'])
        plotter.plot('loss', 'epoch', 'L1', 'Losses', epoch, train_data['L1'])
        plotter.plot('loss', 'epoch', 'L2', 'Losses', epoch, train_data['L2'])

        # Mining stats
        plotter.plot('dualtriplet number', 'epoch', 'num dualtriplet', 'Dual Triplets Mining',
                         epoch, train_data['num_dualtriplets'])
        plotter.plot('dualtriplet number', 'epoch', 'num srctriplet', 'Dual Triplets Mining',
                          epoch, train_data['num_srctriplets'])
        plotter.plot('dualtriplet number', 'epoch', 'num tgttriplet', 'Dual Triplets Mining',
                          epoch, train_data['num_tgttriplets'])
        plotter.plot('num clusters', 'epoch', 'train', 'Number of Clusters',
                          epoch, train_data['n_clusters'])
        plotter.plot('scores', 'epoch', 'train', 'Clustering Scores',
                          epoch, train_data['clustering_scores'])

        # Distance stats
        plotter.plot('distance', 'epoch', 'an', 'Pairwise mean distance',
                          epoch, train_data['dan'])
        plotter.plot('distance', 'epoch', 'ap', 'Pairwise mean distance',
                          epoch, train_data['dap'])

        # Save model
        if not (epoch + 1) % config.output.save_interval:

            model_file_path = os.path.join(model_dir, 'model_{}.pth'.format(epoch))
            print('\nSave model at {}'.format(model_file_path))
            torch.save({'epoch': epoch,
                        'model_state_dict': utils.state_dict_to_cpu(model.state_dict()),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'embedding_size': config.model.embedding_size
                        }, model_file_path)

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