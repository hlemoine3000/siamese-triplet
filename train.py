import sys
import argparse
import configparser

import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
import torchvision.models as models

from dataset_utils import ImageFolder_BalancedBatchSampler, PairsDataset
from models import InceptionResNetV2
from trainer import Triplet_Trainer
import utils

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the configuration file', default='config/train.ini')

    return parser.parse_args(argv)

def main(args):

    print('Triplet loss training.')
    print('CONFIGURATION:\t{}'.format(args.config))
    config = configparser.ConfigParser()
    config.read(args.config)

    # Dataset
    train_dir = config.get('Dataset', 'train_dir')
    test_dir = config.get('Dataset', 'test_dir')
    pairs_file = config.get('Dataset', 'pairs_file')

    # Hyperparameters
    lr = config.getfloat('Hyper Parameters', 'learning_rate')
    weight_decay = config.getfloat('Hyper Parameters', 'weight_decay')
    n_epochs = config.getint('Hyper Parameters', 'n_epochs')

    image_size = config.getint('Hyper Parameters', 'image_size')
    margin = config.getfloat('Hyper Parameters', 'margin')
    people_per_batch = config.getint('Hyper Parameters', 'people_per_batch')
    images_per_person = config.getint('Hyper Parameters', 'images_per_person')
    embedding_size = config.getint('Hyper Parameters', 'embedding_size')
    keep_probability = config.getfloat('Hyper Parameters', 'keep_probability')

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=2),
        transforms.ToTensor()
    ])

    # Train loader
    print('TRAIN SET:\t{}'.format(train_dir))
    train_set = datasets.ImageFolder(train_dir, transform=data_transform)

    batch_sampler = ImageFolder_BalancedBatchSampler(train_set, people_per_batch, images_per_person)
    online_train_loader = torch.utils.data.DataLoader(train_set,
                                                      num_workers=1,
                                                      batch_sampler=batch_sampler,
                                                      pin_memory=True)

    # Test loader
    print('TEST SET:\t{}'.format(test_dir))
    test_set = PairsDataset(test_dir, pairs_file, transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=1, batch_size=(people_per_batch * images_per_person)//2)


    print('Building training graph')
    miner = utils.Triplet_Miner(margin, people_per_batch, images_per_person)

    # model = InceptionResNetV2(bottleneck_layer_size=embedding_size, keep_probability=keep_probability)
    model = models.resnet50(num_classes=embedding_size)
    model.to(device)

    loss = nn.TripletMarginLoss(margin=margin)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, eps=1.0, weight_decay=weight_decay, momentum=0.9, centered=False)

    scheduler = lr_scheduler.StepLR(optimizer, 50, gamma=0.1)

    plotter = utils.VisdomLinePlotter(env_name='Triplet Loss Plots', port=8097)

    trainer = Triplet_Trainer(model,
                              miner,
                              loss,
                              optimizer,
                              scheduler,
                              device,
                              plotter,
                              margin,
                              embedding_size)

    # Loop over epochs
    print('Training Launched.')
    for epoch in range(n_epochs):

        print('Epoch {}'.format(epoch))
        trainer.Train_Epoch(online_train_loader)

        print('Evaluation')
        trainer.Evaluate(test_loader)

    print('Finish.')

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))