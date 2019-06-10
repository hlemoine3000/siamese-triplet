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
from torchvision import transforms

from dataset_utils import dataloaders, vggface2, coxs2v
from ml_utils import trainer, losses, miners
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
    parameters = config.hyperparameters

    # Set up output directory
    # subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    subdir = config.visdom.environment_name
    model_dir = os.path.join(os.path.expanduser(config.output.output_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    else:
        raise Exception('Environment name {} already taken.'.format(subdir))
    config_filename = path_leaf(args.config)
    copyfile(args.config, os.path.join(model_dir, config_filename))

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((parameters["image_size"], parameters["image_size"]), interpolation=1),
        transforms.ToTensor()
    ])

    ###########################
    # SET UP DATALOADERS HERE #
    ###########################

    test_batch_size = (config.hyperparameters.people_per_batch * config.hyperparameters.images_per_person) // 2
    nrof_folds = config.dataset.cross_validation.num_fold
    fold_tool = utils.FoldGenerator(nrof_folds,
                                    config.dataset.cross_validation.num_train_folds,
                                    config.dataset.cross_validation.num_val_folds)
    train_folds, val_folds, test_folds = fold_tool.get_fold()


    # online_train_loader = vggface2.get_vggface2_trainset(config.dataset.vggface2.train_dir,
    #                                                      data_transform,
    #                                                      config.hyperparameters.people_per_batch,
    #                                                      config.hyperparameters.images_per_person)

    online_train_loader = coxs2v.get_coxs2v_trainset(config.dataset.coxs2v.still_dir,
                                                     config.dataset.coxs2v.video3_dir,
                                                     config.dataset.coxs2v.video3_pairs,
                                                     train_folds,
                                                     nrof_folds,
                                                     data_transform,
                                                     config.hyperparameters.people_per_batch,
                                                     config.hyperparameters.images_per_person)

    test_container = dataloaders.Get_TestDataloaders(config,
                                                     data_transform,
                                                     test_batch_size,
                                                     test_folds,
                                                     nrof_folds,
                                                     is_vggface2=False,
                                                     is_lfw=True,
                                                     is_cox_video1=True,
                                                     is_cox_video2=True,
                                                     is_cox_video3=True,
                                                     is_cox_video4=True)

    ##########################
    # SET UP DATALOADERS END #
    ##########################

    #Set up training model
    print('Building training model')
    if config.model.checkpoint:
        print('Loading from checkpoint {}'.format(config.model.checkpoint_path))
        checkpoint = torch.load(config.model.checkpoint_path)
        embedding_size = checkpoint['embedding_size']
        start_epoch = checkpoint['epoch']
    else:
        embedding_size = config.model.embedding_size
        start_epoch = 0

    model = models.load_model(config.model.model_arch,
                              embedding_size=embedding_size,
                              imgnet_pretrained=config.model.pretrained_imagenet)

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=0.9, nesterov=True, weight_decay=2e-4)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, eps=1.0, weight_decay=weight_decay, momentum=0.9, centered=False)

    # scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    scheduler = lr_scheduler.ExponentialLR(optimizer, parameters['learning_rate_decay_factor'])

    if config.model.checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    model = model.to(device)
    # optimizer = optimizer

    plotter = utils.VisdomLinePlotter(env_name=config.visdom.environment_name, port=config.visdom.port)

    # miner = utils.Triplet_Miner(margin, people_per_batch, images_per_person)
    miner = miners.SemihardNegativeTripletSelector(config.hyperparameters.margin)

    # loss = nn.TripletMarginLoss(margin=parameters['margin'], swap=parameters['triplet_swap'])
    # loss = utils.TripletLoss(margin=parameters['margin'])
    loss = losses.TripletLoss(config.hyperparameters.margin)

    my_trainer = trainer.Triplet_Trainer(model,
                                         miner,
                                         loss,
                                         optimizer,
                                         scheduler,
                                         device,
                                         plotter,
                                         config.hyperparameters.margin,
                                         config.model.embedding_size,
                                         config.visdom.log_interval)

    # Loop over epochs
    print('Training Launched.')
    for epoch in range(start_epoch, parameters['n_epochs']):

        # Validation
        for test_name, test_loader in test_container:
            print('\nEvaluation on {}'.format(test_name))
            my_trainer.Evaluate(test_loader,
                                name=test_name,
                                nrof_folds=nrof_folds)

        # Training
        print('\nTrain Epoch {}'.format(epoch))
        my_trainer.Train_Epoch(online_train_loader)

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

    print('Finish.')

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))