
import torch
import tqdm
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import utils
from evaluation import Evaluate


def get_trainer(mode,
                model: nn.Module,
                miner,
                loss: _Loss,
                optimizer: Optimizer,
                scheduler: _LRScheduler,
                device,
                plotter: utils.VisdomLinePlotter,
                margin: int,
                embedding_size: int,
                log_interval: int=1):

    if mode == 'dualtriplet':
        return Dualtriplet_Trainer(model,
                                   miner,
                                   loss,
                                   optimizer,
                                   scheduler,
                                   device,
                                   plotter,
                                   margin,
                                   embedding_size,
                                   log_interval)

    elif mode == 'supervised_dualtriplet':
        return Supervised_Dualtriplet_Trainer(model,
                                              miner,
                                              loss,
                                              optimizer,
                                              scheduler,
                                              device,
                                              margin)

    else:
        raise Exception('Trainer type {} does not exist.'.format(mode))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Triplet_Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 miner,
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
        self.loss = loss

    def Train_Epoch(self,
                    train_loader: DataLoader):

        self.model.train()
        train_losses = utils.AverageMeter()
        # train_ap_distances = utils.AverageMeter()
        # train_an_distances = utils.AverageMeter()
        train_triplets = utils.AverageMeter()

        # Training
        self.scheduler.step()
        lr = get_lr(self.optimizer)
        # self.plotter.plot('learning rate', 'step', 'train', 'Learning Rate',
        #                   self.step, lr)

        triplets_list = []
        loader_length = len(train_loader)
        tbar = tqdm.tqdm(train_loader)
        step = 0
        for i, (local_batch, local_labels) in enumerate(tbar):
            # Transfer to GPU
            local_batch = local_batch.to(self.device)
            embeddings = self.model.forward(local_batch)

            triplets = self.miner.get_triplets(embeddings.cpu(), local_labels)
            triplets_list.append(triplets)
            triplets_train = torch.cat(triplets_list)

            if len(triplets_train) > 200:

                a = embeddings[triplets_train[:, 0]]
                p = embeddings[triplets_train[:, 1]]
                n = embeddings[triplets_train[:, 2]]

                loss = self.loss(a, p, n)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # tbar.set_description('Step {} - Loss: {:.4f}'.format(self.step, loss.item()))

                # ap_distances = torch.norm(a - p, p=2, dim=1)
                # an_distances = torch.norm(a - n, p=2, dim=1)

                train_losses.append(loss.item())
                # train_ap_distances.append(ap_distances.mean().item())
                # train_an_distances.append(an_distances.mean().item())
                train_triplets.append(triplets_train.size(0))

                step += 1
                triplets_list = []

        if step == 0:
            print('Not enough triplets.')
            return None
        else:
            data_dict = {'loss': train_losses.last_avg,
                         'num_triplets': train_triplets.last_avg,
                         'lr': lr}

            return data_dict

    def Evaluate(self,
                 test_loader: DataLoader,
                 name='validation',
                 nrof_folds=10,
                 distance_metric=0,
                 val_far=1e-3):

        return Evaluate(test_loader,
                        self.model,
                        self.device,
                        self.step,
                        plotter=self.plotter,
                        name=name,
                        nrof_folds=nrof_folds,
                        distance_metric=distance_metric,
                        val_far=val_far)


class Dualtriplet_Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 miner,
                 loss: _Loss,
                 optimizer: Optimizer,
                 scheduler: _LRScheduler,
                 device,
                 plotter: utils.VisdomLinePlotter,
                 margin: int,
                 embedding_size: int,
                 log_interval: int = 1,
                 ):

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
        self.loss = loss

    def Train_Epoch(self,
                    source_loader: DataLoader,
                    target_loader: DataLoader):

        self.model.train()

        log_loss_dict = {}
        train_losses = utils.AverageMeter()
        for loss_key in self.loss.loss_keys:
            log_loss_dict[loss_key] = utils.AverageMeter()

        train_ap_distances = utils.AverageMeter()
        train_an_distances = utils.AverageMeter()
        num_dualtriplets = utils.AverageMeter()
        num_srctriplets = utils.AverageMeter()
        num_tgttriplets = utils.AverageMeter()
        n_clusters = utils.AverageMeter()
        clustering_scores = utils.AverageMeter()

        # Training
        self.scheduler.step()
        lr = get_lr(self.optimizer)
        # self.plotter.plot('learning rate', 'step', 'train', 'Learning Rate',
        #                   self.step, lr)

        step = 0
        data_loader = zip(source_loader, target_loader)
        tbar = tqdm.tqdm(data_loader)
        dual_triplets_list = []
        for i, ((source_batch, source_labels), (target_batch, target_labels)) in enumerate(tbar):
            # Forward on source
            source_batch = source_batch.to(self.device)
            source_embeddings = self.model.forward(source_batch)

            # Forward on target
            target_batch = target_batch.to(self.device)
            target_embeddings = self.model.forward(target_batch)

            dual_triplets, miner_dict = self.miner.get_dualtriplet(source_embeddings, source_labels, target_embeddings)
            dual_triplets_list.append(dual_triplets)
            dual_triplets_train = torch.cat(dual_triplets_list)

            if len(dual_triplets_train) > 100:

                a = source_embeddings[dual_triplets_train[:, 0]]
                p = source_embeddings[dual_triplets_train[:, 1]]
                n = source_embeddings[dual_triplets_train[:, 2]]
                ta = target_embeddings[dual_triplets_train[:, 3]]
                tp = target_embeddings[dual_triplets_train[:, 4]]
                tn = target_embeddings[dual_triplets_train[:, 5]]

                loss, losses_dict = self.loss(a, p, n, ta, tp, tn)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                ap_distances = torch.norm(a - p, p=2, dim=1)
                an_distances = torch.norm(a - n, p=2, dim=1)

                train_losses.append(loss.item())
                for loss_key in self.loss.loss_keys:
                    log_loss_dict[loss_key].append(losses_dict[loss_key].item())

                train_ap_distances.append(ap_distances.mean().item())
                train_an_distances.append(an_distances.mean().item())

                num_dualtriplets.append(len(dual_triplets_train))
                num_srctriplets.append(miner_dict['num_src_triplets'])
                num_tgttriplets.append(miner_dict['num_tgt_triplets'])
                n_clusters.append(miner_dict['n_clusters'])
                clustering_scores.append(miner_dict['clustering_score'])

                # if not (i + 1) % self.log_interval or (i + 1) == loader_length:
                #
                #     # Loss stats
                #     self.plotter.plot('loss', 'step', 'train', 'Loss', self.step, train_losses.last_avg)
                #     for loss_key in self.loss.loss_keys:
                #         self.plotter.plot('loss', 'step', loss_key, 'Loss', self.step, log_loss_dict[loss_key].last_avg)
                #
                #     # Mining stats
                #     self.plotter.plot('dualtriplet number', 'step', 'num dualtriplet', 'Dual Triplets Mining',
                #                      self.step, num_dualtriplets.last_avg)
                #     self.plotter.plot('dualtriplet number', 'step', 'num srctriplet', 'Dual Triplets Mining',
                #                       self.step, num_srctriplets.last_avg)
                #     self.plotter.plot('dualtriplet number', 'step', 'num tgttriplet', 'Dual Triplets Mining',
                #                       self.step, num_tgttriplets.last_avg)
                #     self.plotter.plot('num clusters', 'step', 'train', 'Number of Clusters',
                #                       self.step, n_clusters.last_avg)
                #     self.plotter.plot('scores', 'step', 'train', 'Clustering Scores',
                #                       self.step, clustering_scores.last_avg)
                #
                #     # Distance stats
                #     self.plotter.plot('distance', 'step', 'an', 'Pairwise mean distance',
                #                       self.step, train_an_distances.last_avg)
                #     self.plotter.plot('distance', 'step', 'ap', 'Pairwise mean distance',
                #                       self.step, train_ap_distances.last_avg)

                step += 1
                dual_triplets_list = []
        # tbar.set_description('Step {} - Loss: {:.4f}'.format(self.step, train_losses.avg))

        if step == 0:
            return None
        else:
            data_dict = {'L12': train_losses.last_avg,
                         'num_dualtriplets': num_dualtriplets.last_avg,
                         'num_srctriplets': num_srctriplets.last_avg,
                         'num_tgttriplets': num_tgttriplets.last_avg,
                         'n_clusters': n_clusters.last_avg,
                         'clustering_scores': clustering_scores.last_avg,
                         'dap': train_ap_distances.last_avg,
                         'dan': train_an_distances.last_avg,
                         'lr': lr}

            for loss_key in self.loss.loss_keys:
                data_dict[loss_key] = log_loss_dict[loss_key].last_avg

            return data_dict


    def Evaluate(self,
                 test_loader: DataLoader,
                 name='validation',
                 nrof_folds=10,
                 distance_metric=0,
                 val_far=1e-3):

        return Evaluate(test_loader,
                        self.model,
                        self.device,
                        self.step,
                        plotter=self.plotter,
                        name=name,
                        nrof_folds=nrof_folds,
                        distance_metric=distance_metric,
                        val_far=val_far)


class Supervised_Dualtriplet_Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 miner,
                 loss: _Loss,
                 optimizer: Optimizer,
                 scheduler: _LRScheduler,
                 device,
                 margin: int):

        self.model = model
        self.miner = miner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.margin = margin
        self.loss = loss

        self.device = device

    def Train_Epoch(self,
                    source_loader: DataLoader,
                    target_loader: DataLoader):

        self.model.train()

        log_loss_dict = {}
        train_losses = utils.AverageMeter()
        for loss_key in self.loss.loss_keys:
            log_loss_dict[loss_key] = utils.AverageMeter()

        train_ap_distances = utils.AverageMeter()
        train_an_distances = utils.AverageMeter()
        num_dualtriplets = utils.AverageMeter()
        num_srctriplets = utils.AverageMeter()
        num_tgttriplets = utils.AverageMeter()

        # Training
        self.scheduler.step()
        lr = get_lr(self.optimizer)

        step = 0
        data_loader = zip(source_loader, target_loader)
        tbar = tqdm.tqdm(data_loader)
        dual_triplets_list = []
        for i, ((source_batch, source_labels), (target_batch, target_labels)) in enumerate(tbar):
            # Forward on source
            source_batch = source_batch.to(self.device)
            source_embeddings = self.model.forward(source_batch)

            # Forward on target
            target_batch = target_batch.to(self.device)
            target_embeddings = self.model.forward(target_batch)

            dual_triplets, miner_dict = self.miner.get_dualtriplet(source_embeddings,
                                                                   source_labels,
                                                                   target_embeddings,
                                                                   target_labels)
            dual_triplets_list.append(dual_triplets)
            dual_triplets_train = torch.cat(dual_triplets_list)

            if len(dual_triplets_train) > 200:

                a = source_embeddings[dual_triplets_train[:, 0]]
                p = source_embeddings[dual_triplets_train[:, 1]]
                n = source_embeddings[dual_triplets_train[:, 2]]
                ta = target_embeddings[dual_triplets_train[:, 3]]
                tp = target_embeddings[dual_triplets_train[:, 4]]
                tn = target_embeddings[dual_triplets_train[:, 5]]

                loss, losses_dict = self.loss(a, p, n, ta, tp, tn)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                ap_distances = torch.norm(a - p, p=2, dim=1)
                an_distances = torch.norm(a - n, p=2, dim=1)

                train_losses.append(loss.item())
                for loss_key in self.loss.loss_keys:
                    log_loss_dict[loss_key].append(losses_dict[loss_key].item())

                train_ap_distances.append(ap_distances.mean().item())
                train_an_distances.append(an_distances.mean().item())

                num_dualtriplets.append(len(dual_triplets_train))
                num_srctriplets.append(miner_dict['num_src_triplets'])
                num_tgttriplets.append(miner_dict['num_tgt_triplets'])

                step += 1
                dual_triplets_list = []

        if step == 0:
            return None
        else:
            data_dict = {'L12': train_losses.last_avg,
                         'num_dualtriplets': num_dualtriplets.last_avg,
                         'num_srctriplets': num_srctriplets.last_avg,
                         'num_tgttriplets': num_tgttriplets.last_avg,
                         'n_clusters': 0,
                         'clustering_scores': 0,
                         'dap': train_ap_distances.last_avg,
                         'dan': train_an_distances.last_avg,
                         'lr': lr}

            for loss_key in self.loss.loss_keys:
                data_dict[loss_key] = log_loss_dict[loss_key].last_avg

            return data_dict


    def Evaluate(self,
                 test_loader: DataLoader,
                 name='validation',
                 nrof_folds=10,
                 distance_metric=0,
                 val_far=1e-3):

        return Evaluate(test_loader,
                        self.model,
                        self.device,
                        0,
                        name=name,
                        nrof_folds=nrof_folds,
                        distance_metric=distance_metric,
                        val_far=val_far)
