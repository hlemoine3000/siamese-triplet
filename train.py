import sys
import argparse
# import configparser
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
    # config = configparser.ConfigParser()
    # config.read(args.config)

    # Dataset
    # train_dir = config.get('Dataset', 'train_dir')
    # test_dir = config.get('Dataset', 'test_dir')
    # pairs_file = config.get('Dataset', 'pairs_file')
    #
    # # Hyperparameters
    # lr = config.getfloat('Hyper Parameters', 'learning_rate')
    # weight_decay = config.getfloat('Hyper Parameters', 'weight_decay')
    # n_epochs = config.getint('Hyper Parameters', 'n_epochs')
    #
    # image_size = config.getint('Hyper Parameters', 'image_size')
    # margin = config.getfloat('Hyper Parameters', 'margin')
    # people_per_batch = config.getint('Hyper Parameters', 'people_per_batch')
    # images_per_person = config.getint('Hyper Parameters', 'images_per_person')
    # embedding_size = config.getint('Hyper Parameters', 'embedding_size')
    #
    # log_interval = config.getint('Hyper Parameters', 'log_interval')

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(parameters['image_size'], interpolation=1),#1
        transforms.ToTensor()
    ])

    # Train loader
    print('TRAIN SET:\t{}'.format(datasets_path['train_dir']))
    train_set = datasets.ImageFolder(datasets_path['train_dir'], transform=data_transform)

    batch_sampler = Random_BalancedBatchSampler(train_set, parameters['people_per_batch'], parameters['images_per_person'], max_batches=1000)
    online_train_loader = torch.utils.data.DataLoader(train_set,
                                                      num_workers=8,
                                                      batch_sampler=batch_sampler,
                                                      pin_memory=True)

    # Test loader
    print('TEST SET:\t{}'.format(datasets_path['test_dir'][0]))
    test_set = PairsDataset(datasets_path['test_dir'][0], datasets_path['pairs_file'][0], transform=data_transform, preload=True)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=2, batch_size=(parameters['people_per_batch'] * parameters['images_per_person'])//2)


    print('Building training model')
    # miner = utils.Triplet_Miner(margin, people_per_batch, images_per_person)
    miner = utils.SemihardNegativeTripletSelector(parameters['margin'])

    m = models.InceptionResNetV2(bottleneck_layer_size=parameters['embedding_size'])
    # m = models.resnet18(num_classes=embedding_size, pretrained=True)
    class NormWrapper(nn.Module):
        def __init__(self, model):
            super(NormWrapper, self).__init__()
            self.model = model
        def forward(self, input):
            embeddings = self.model(input)
            norms = embeddings.pow(2).sum(1, keepdim=True).add(1e-8).sqrt()
            return embeddings / norms
    model = NormWrapper(model=m)
    model.to(device)

    loss = nn.TripletMarginLoss(margin=parameters['margin'])
    # loss = utils.TripletLoss(margin=margin)

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=0.9, nesterov=True, weight_decay=2e-4)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, eps=1.0, weight_decay=weight_decay, momentum=0.9, centered=False)

    # scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    scheduler = lr_scheduler.ExponentialLR(optimizer, parameters['learning_rate_decay_factor'])

    plotter = utils.VisdomLinePlotter(env_name='Triplet Loss Plots 2', port=8097)

    trainer = Triplet_Trainer(model,
                              miner,
                              loss,
                              optimizer,
                              scheduler,
                              device,
                              plotter,
                              parameters['margin'],
                              parameters['embedding_size'],
                              parameters['log_interval'])

    # Loop over epochs
    print('Training Launched.')
    for epoch in range(parameters['n_epochs']):

        print('Epoch {}'.format(epoch))
        trainer.Train_Epoch(online_train_loader)

        print('Evaluation')
        trainer.Evaluate(test_loader)

    print('Finish.')

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))