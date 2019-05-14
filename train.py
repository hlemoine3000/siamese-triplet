import sys
import os
import argparse
from datetime import datetime
import json

import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torch.utils import data

from dataset_utils import ImageFolder_BalancedBatchSampler, PairsDataset, Random_BalancedBatchSampler
import models
from trainer import Triplet_Trainer
import utils

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/train_config.json')

    return parser.parse_args(argv)

def main(args):

    print('Triplet loss training.')
    print('CONFIGURATION:\t{}'.format(args.config))
    with open(args.config) as json_config_file:
        config = json.load(json_config_file)
    parameters = config['hyperparameters']
    datasets_path = config['dataset']
    visdom_config = config['visdom']
    output_config = config['output']

    # Set up output directory
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = os.path.join(os.path.expanduser(output_config['output_dir']), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((parameters['image_size'], parameters['image_size']), interpolation=1),
        transforms.ToTensor()
    ])

    # Train loader
    print('TRAIN SET {}:\t{}'.format(datasets_path['train_name'], datasets_path['train_dir']))
    train_set = datasets.ImageFolder(datasets_path['train_dir'], transform=data_transform)

    batch_sampler = Random_BalancedBatchSampler(train_set, parameters['people_per_batch'], parameters['images_per_person'], max_batches=1000)
    online_train_loader = torch.utils.data.DataLoader(train_set,
                                                      num_workers=8,
                                                      batch_sampler=batch_sampler,
                                                      pin_memory=True)

    test_container = []
    for i, test_name in enumerate(datasets_path['test_name']):
        # Test loader
        print('TEST SET {}:\t{}'.format(test_name, datasets_path['test_dir'][i]))
        test_set = PairsDataset(datasets_path['test_dir'][i], datasets_path['pairs_file'][i], transform=data_transform, preload=True)
        test_loader = torch.utils.data.DataLoader(test_set, num_workers=2, batch_size=(parameters['people_per_batch'] * parameters['images_per_person'])//2)
        test_container.append((test_name, test_loader))

    print('Building training model')
    # miner = utils.Triplet_Miner(margin, people_per_batch, images_per_person)
    miner = utils.SemihardNegativeTripletSelector(parameters['margin'])

    # model = models.InceptionResNetV2(bottleneck_layer_size=parameters['embedding_size'])
    model = models.resnet18(num_classes=parameters['embedding_size'], pretrained=True)
    model.to(device)

    loss = nn.TripletMarginLoss(margin=parameters['margin'], swap=parameters['triplet_swap'])
    # loss = utils.TripletLoss(margin=margin)

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=0.9, nesterov=True, weight_decay=2e-4)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, eps=1.0, weight_decay=weight_decay, momentum=0.9, centered=False)

    # scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    scheduler = lr_scheduler.ExponentialLR(optimizer, parameters['learning_rate_decay_factor'])

    plotter = utils.VisdomLinePlotter(env_name=visdom_config['environment_name'], port=8097)

    trainer = Triplet_Trainer(model,
                              miner,
                              loss,
                              optimizer,
                              scheduler,
                              device,
                              plotter,
                              parameters['margin'],
                              parameters['embedding_size'],
                              visdom_config['log_interval'])

    # Loop over epochs
    print('Training Launched.')
    for epoch in range(parameters['n_epochs']):

        print('Train Epoch {}'.format(epoch))
        trainer.Train_Epoch(online_train_loader)

        for test_name, test_loader in test_container:
            print('Evaluation on {}'.format(test_name))
            trainer.Evaluate(test_loader, name=test_name)

        if not (epoch + 1) % output_config['save_interval']:
            print('Save model at {}'.format(model_dir))
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

    print('Finish.')

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))