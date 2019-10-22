
import torch
import tqdm
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import utils
from ml_utils import ml_utils, clustering


class Triplet_Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 miner,
                 loss: _Loss,
                 optimizer: Optimizer,
                 scheduler: _LRScheduler,
                 device,
                 plotter: utils.VisdomPlotter,
                 margin: int,
                 embedding_size: int,
                 eval_function,
                 batch_size: int = 32):

        self.model = model
        self.miner = miner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.plotter = plotter
        self.margin = margin
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.eval_function = eval_function
        self.device = device

        self.step = 0
        self.loss = loss

    def Train_Epoch(self,
                    train_loader: DataLoader,
                    epoch: int):

        self.model.train()
        data_dict = utils.AverageData_Dict()
        num_triplets = 0

        # Training
        source_anchor_images = []
        source_positive_images = []
        source_negative_images = []

        training_step = 0
        tbar = tqdm.tqdm(train_loader)
        for i, (batch, labels) in enumerate(tbar):
            batch = batch.to(self.device)

            self.model.eval()
            with torch.no_grad():
                embeddings = self.model(batch)

            triplets_indexes = self.miner.get_triplets(embeddings, labels)
            num_triplets += len(triplets_indexes)
            data_dict['triplets_per_step'].append(len(triplets_indexes))

            source_anchor_images.append(batch[triplets_indexes[:, 0]])
            source_positive_images.append(batch[triplets_indexes[:, 1]])
            source_negative_images.append(batch[triplets_indexes[:, 2]])

            if num_triplets >= self.batch_size:

                source_anchor_images = torch.cat(source_anchor_images, 0)
                source_positive_images = torch.cat(source_positive_images, 0)
                source_negative_images = torch.cat(source_negative_images, 0)

                self.optimizer.zero_grad()
                nb_chunks = num_triplets // self.batch_size
                for chunk_idx in range(nb_chunks):

                    lower_idx = chunk_idx * self.batch_size
                    higher_idx = chunk_idx * self.batch_size + self.batch_size

                    self.model.train()
                    image_tensor = torch.cat([source_anchor_images[lower_idx:higher_idx],
                                              source_positive_images[lower_idx:higher_idx],
                                              source_negative_images[lower_idx:higher_idx]], 0)
                    embeddings = self.model(image_tensor)
                    embeddings_list = torch.chunk(embeddings, 3, 0)

                    loss = self.loss(*embeddings_list)# / nb_chunks
                    loss.backward()

                    ap_distances = torch.norm(embeddings_list[0] - embeddings_list[1], p=2, dim=1)
                    an_distances = torch.norm(embeddings_list[0] - embeddings_list[2], p=2, dim=1)

                    data_dict['loss'].append(loss.item())
                    data_dict['dap'].append(ap_distances.mean().item())
                    data_dict['dan'].append(an_distances.mean().item())

                    training_step += 1
                    # tbar.set_postfix({'training steps': training_step})

                self.optimizer.step()
                num_triplets = 0
                source_anchor_images = []
                source_positive_images = []
                source_negative_images = []

        lr = ml_utils.get_lr(self.optimizer)
        self.scheduler.step()

        self.plotter.plot('learning rate', 'epoch', 'train', 'Learning Rate',
                          epoch, lr)
        self.plotter.plot('triplet number', 'epoch', 'triplets per step', 'Triplets Mining',
                          epoch, data_dict['triplets_per_step'].last_avg())

        if training_step > 0:
            self.plotter.plot('loss', 'epoch', 'train_loss', 'Losses', epoch, data_dict['loss'].last_avg())
            self.plotter.plot('distance', 'epoch', 'train_an', 'Pairwise mean distance',
                              epoch, data_dict['dan'].last_avg())
            self.plotter.plot('distance', 'epoch', 'train_ap', 'Pairwise mean distance',
                              epoch, data_dict['dap'].last_avg())

            self.miner.plot(epoch)


class Dualtriplet_Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 miner,
                 loss: _Loss,
                 optimizer: Optimizer,
                 scheduler: _LRScheduler,
                 device,
                 plotter: utils.VisdomPlotter,
                 margin: int,
                 embedding_size: int,
                 batch_size: int = 32):

        self.model = model
        self.miner = miner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.plotter = plotter
        self.margin = margin
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.device = device

        self.step = 0
        self.loss = loss

    def Train_Epoch(self,
                    source_loader: DataLoader,
                    target_loader: DataLoader,
                    epoch):

        self.model.train()
        data_dict = utils.AverageData_Dict()
        num_dualtriplets = 0
        total_dualtriplets = 0

        # Training
        source_anchor_images = []
        source_positive_images = []
        source_negatives_images = []
        target_anchor_images = []
        target_positive_images = []
        target_negatives_images = []

        clustering.update_gaussian_mixture(self.miner.gmixture,
                                           target_loader,
                                           self.model,
                                           self.device,
                                           _plotter=self.plotter,
                                           name='Target Gaussians')

        training_step = 0
        data_loader = zip(source_loader, target_loader)
        tbar = tqdm.tqdm(data_loader, total=min(len(source_loader), len(target_loader)))
        for i, ((source_batch, source_labels), (target_batch, target_labels)) in enumerate(tbar):
            source_batch, target_batch = source_batch.to(self.device), target_batch.to(self.device)
            batch = torch.cat([source_batch, target_batch], 0)

            self.model.eval()
            with torch.no_grad():
                embeddings = self.model(batch)
                source_embeddings, target_embeddings = torch.chunk(embeddings, 2, 0)

            dualtriplets_indexes = self.miner.get_dualtriplet(source_embeddings, source_labels, target_embeddings, target_labels)
            num_dualtriplets += len(dualtriplets_indexes)

            source_anchor_images.append(source_batch[dualtriplets_indexes[:, 0]])
            source_positive_images.append(source_batch[dualtriplets_indexes[:, 1]])
            source_negatives_images.append(source_batch[dualtriplets_indexes[:, 2]])
            target_anchor_images.append(target_batch[dualtriplets_indexes[:, 3]])
            target_positive_images.append(target_batch[dualtriplets_indexes[:, 4]])
            target_negatives_images.append(target_batch[dualtriplets_indexes[:, 5]])

            if num_dualtriplets >= self.batch_size:
                source_anchor_images = torch.cat(source_anchor_images, 0)[:self.batch_size]
                source_positive_images = torch.cat(source_positive_images, 0)[:self.batch_size]
                source_negatives_images = torch.cat(source_negatives_images, 0)[:self.batch_size]
                target_anchor_images = torch.cat(target_anchor_images, 0)[:self.batch_size]
                target_positive_images = torch.cat(target_positive_images, 0)[:self.batch_size]
                target_negatives_images = torch.cat(target_negatives_images, 0)[:self.batch_size]

                self.model.train()
                image_tensor = torch.cat([source_anchor_images, source_positive_images, source_negatives_images,
                                          target_anchor_images, target_positive_images, target_negatives_images], 0)
                embeddings = self.model(image_tensor)
                embeddings_list = torch.chunk(embeddings, 6, 0)

                loss = self.loss(*embeddings_list)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                src_ap_distances = torch.norm(embeddings_list[0] - embeddings_list[1], p=2, dim=1)
                src_an_distances = torch.norm(embeddings_list[0] - embeddings_list[2], p=2, dim=1)
                tgt_ap_distances = torch.norm(embeddings_list[3] - embeddings_list[4], p=2, dim=1)
                tgt_an_distances = torch.norm(embeddings_list[3] - embeddings_list[5], p=2, dim=1)

                data_dict['src_dap'].append(src_ap_distances.mean().item())
                data_dict['src_dan'].append(src_an_distances.mean().item())
                data_dict['tgt_dap'].append(tgt_ap_distances.mean().item())
                data_dict['tgt_dan'].append(tgt_an_distances.mean().item())

                num_dualtriplets = 0
                source_anchor_images = []
                source_positive_images = []
                source_negatives_images = []
                target_anchor_images = []
                target_positive_images = []
                target_negatives_images = []
                training_step += 1
                total_dualtriplets += self.batch_size
                tbar.set_postfix({'training steps': training_step})

                self.loss.plot(self.step)
                self.miner.plot(self.step)
                self.step += 1

                continue

        lr = ml_utils.get_lr(self.optimizer)
        self.scheduler.step()

        # self.plotter.plot('learning rate', 'epoch', 'train', 'Learning Rate',
        #              epoch, lr)
        # self.plotter.plot('dualtriplet number', 'epoch', 'total dualtriplets', 'Dual Triplets Mining',
        #                   epoch, total_dualtriplets)
        #
        # if training_step > 0:
        #
        #     self.loss.plot(epoch)
        #     self.miner.plot(epoch)
