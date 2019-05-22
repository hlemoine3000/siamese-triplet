from copy import deepcopy

import torch
import numpy as np
import tqdm
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import pandas as pd
import utils

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Triplet_Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 miner: utils.Triplet_Miner,
                 loss: _Loss,
                 optimizer: Optimizer,
                 scheduler: _LRScheduler,
                 device,
                 plotter: utils.VisdomLinePlotter,
                 margin: int,
                 embedding_size: int,
                 log_interval: int=1):

        self.model = model
        self.miner = miner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.plotter = plotter
        self.margin = margin
        self.embedding_size = embedding_size

        self.device = device
        self.log_interval = log_interval

        self.step = 0
        # self.loss_function = utils.TripletLoss(self.margin)
        self.loss = loss

    def Train_Epoch(self,
                    train_loader: DataLoader):

        self.model.train()
        train_losses = utils.AverageMeter()
        train_ap_distances = utils.AverageMeter()
        train_an_distances = utils.AverageMeter()
        train_triplets = utils.AverageMeter()

        # Training
        self.scheduler.step()
        lr = get_lr(self.optimizer)
        self.plotter.plot('learning rate', 'step', 'train', 'Learning Rate',
                          self.step, lr)

        loader_length = len(train_loader)
        tbar = tqdm.tqdm(train_loader, dynamic_ncols=True)
        for i, (local_batch, local_labels) in enumerate(tbar):
            # Transfer to GPU
            local_batch = local_batch.to(self.device)
            embeddings = self.model.forward(local_batch)

            triplets = self.miner.get_triplets(embeddings.cpu(), local_labels)

            a = embeddings[triplets[:, 0]]
            p = embeddings[triplets[:, 1]]
            n = embeddings[triplets[:, 2]]

            # loss = F.triplet_margin_loss(a, p, n, margin=self.margin)
            loss = self.loss(a, p, n)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tbar.set_description('Step {} - Loss: {:.4f}'.format(self.step, loss.item()))

            ap_distances = torch.norm(embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]], p=2, dim=1)
            an_distances = torch.norm(embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]], p=2, dim=1)

            train_losses.append(loss.item())
            train_ap_distances.append(ap_distances.mean().item())
            train_an_distances.append(an_distances.mean().item())
            train_triplets.append(triplets.size(0))

            if not (i + 1) % self.log_interval or (i + 1) == loader_length:
                self.plotter.plot('distance', 'step', 'vggface2_an', 'Pairwise mean distance',
                                  self.step, train_an_distances.last_avg)
                self.plotter.plot('distance', 'step', 'vggface2_ap', 'Pairwise mean distance',
                                  self.step, train_ap_distances.last_avg)
                self.plotter.plot('loss', 'step', 'train', 'Triplet Loss', self.step, train_losses.last_avg)
                self.plotter.plot('triplet number', 'step', 'train', 'Triplet Mining',
                                  self.step, train_triplets.last_avg)

            self.step += 1
        tbar.set_description('Step {} - Loss: {:.4f}'.format(self.step, train_losses.avg))

    def Evaluate(self,
                 test_loader: DataLoader,
                 name='validation',
                 nrof_folds=10,
                 distance_metric=0):

        embeddings1 = []
        embeddings2 = []
        issame_array =[]

        self.model.eval()

        with torch.no_grad():
            tbar = tqdm.tqdm(test_loader, dynamic_ncols=True)
            for images_batch, issame, path_batch in tbar:
                # Transfer to GPU

                image_batch1 = images_batch[0].to(self.device, non_blocking=True)
                image_batch2 = images_batch[1].to(self.device, non_blocking=True)

                emb1 = self.model.forward(image_batch1)
                emb2 = self.model.forward(image_batch2)

                embeddings1.append(emb1)
                embeddings2.append(emb2)
                issame_array.append(deepcopy(issame))

            embeddings1 = torch.cat(embeddings1, 0).cpu().numpy()
            embeddings2 = torch.cat(embeddings2, 0).cpu().numpy()
            issame_array = torch.cat(issame_array, 0).cpu().numpy()

        distance_and_is_same = zip(np.sum((embeddings1 - embeddings2)**2, axis=1), issame_array)
        distance_and_is_same_df = pd.DataFrame(distance_and_is_same)
        negative_mean_distance = distance_and_is_same_df[distance_and_is_same_df[1] == False][0].mean()
        positive_mean_distance = distance_and_is_same_df[distance_and_is_same_df[1] == True][0].mean()

        thresholds = np.arange(0, 4, 0.01)
        subtract_mean = False

        tpr, fpr, accuracy, best_threshold = utils.Calculate_Roc(thresholds, embeddings1, embeddings2,
                                                                   np.asarray(issame_array), nrof_folds=nrof_folds,
                                                                   distance_metric=distance_metric,
                                                                   subtract_mean=subtract_mean)

        thresholds = np.arange(0, 4, 0.001)
        val, val_std, far, threshold_lowfar = utils.Calculate_Val(thresholds, embeddings1, embeddings2,
                                                                    np.asarray(issame_array), 1e-3,
                                                                    nrof_folds=nrof_folds,
                                                                    distance_metric=distance_metric,
                                                                    subtract_mean=subtract_mean)

        print('Accuracy: {:.3%}+-{:.3%}'.format(np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: {:.3%}+-{:.3%} @ FAR={:.3%}'.format(val, val_std, far))
        print('Best threshold: {:.3f}'.format(best_threshold))

        self.plotter.plot('distance', 'step', name + '_an', 'Pairwise mean distance', self.step, negative_mean_distance)
        self.plotter.plot('distance', 'step', name + '_ap', 'Pairwise mean distance', self.step, positive_mean_distance)

        self.plotter.plot('accuracy', 'step', name, 'Accuracy', self.step, np.mean(accuracy))
        self.plotter.plot('validation rate', 'step', name, 'Validation Rate', self.step, val)
        self.plotter.plot('best threshold', 'step', name, 'Best Threshold', self.step, best_threshold)

            # return tpr, fpr, accuracy, val, val_std, far, best_threshold, threshold_lowfar, tpr_lowfar, acc_lowfar

# def fit(train_loader: DataLoader,
#         val_loader: DataLoader,
#         model: nn.Module,
#         loss_fn: nn.Module,
#         optimizer: Optimizer,
#         scheduler: _LRScheduler,
#         n_epochs,
#         cuda,
#         log_interval,
#         metrics=[],
#         start_epoch=0):
#     """
#     Loaders, model, loss function and metrics should work together for a given task,
#     i.e. The model should be able to process data output of loaders,
#     loss function should process target output of loaders and outputs from the model
#
#     Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
#     Siamese network: Siamese loader, siamese model, contrastive loss
#     Online triplet learning: batch loader, embedding model, online triplet loss
#     """
#     for epoch in range(0, start_epoch):
#         scheduler.step()
#
#     for epoch in range(start_epoch, n_epochs):
#         scheduler.step()
#
#         # Train stage
#         train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
#
#         message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
#         for metric in metrics:
#             message += '\t{}: {}'.format(metric.name(), metric.value())
#
#         # val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
#         # val_loss /= len(val_loader)
#         #
#         # message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
#         #                                                                          val_loss)
#         # for metric in metrics:
#         #     message += '\t{}: {}'.format(metric.name(), metric.value())
#
#         print(message)

# def train_epoch(train_loader: DataLoader,
#                 model: nn.Module,
#                 loss_fn: nn.Module,
#                 optimizer: Optimizer,
#                 device,
#                 log_interval,
#                 metrics):
#
#     for metric in metrics:
#         metric.reset()
#
#     model.train()
#     losses = []
#     total_loss = 0
#
#     batch_idx = 0
#     for local_batch, local_labels in train_loader:
#         # Transfer to GPU
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)
#
#         optimizer.zero_grad()
#
#         embedddings = model.forward(local_batch)
#         loss, num_triplets = loss_fn(embedddings, local_labels)
#
#         losses.append(loss)
#         total_loss += loss
#         loss.backward()
#         optimizer.step()
#
#         for metric in metrics:
#             metric(embedddings, local_labels, loss)
#
#
#         if batch_idx % log_interval == 0:
#             message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 batch_idx * len(embedddings), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), torch.mean(torch.stack(losses)))
#             for metric in metrics:
#                 message += '\t{}: {}'.format(metric.name(), metric.value())
#
#             print(message)
#             losses = []
#
#         batch_idx += 1
#
#     total_loss /= (batch_idx + 1)
#     return total_loss, metrics

def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
