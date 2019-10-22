
from itertools import combinations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import torch

import utils
from ml_utils import ml_utils, miner_utils


def get_miner(mode, margin, people_per_batch, plotter, deadzone_ratio=0.1):

    if mode == 'dualtriplet':
        return FunctionDualTripletSelector3(margin,
                                            plotter,
                                            deadzone_ratio=deadzone_ratio)

    elif mode == 'supervised_dualtriplet':
        return FunctionDualTripletSupervisedSelector(margin, plotter)

    else:
        raise Exception('Miner type {} does not exist.'.format(mode))


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = ml_utils.pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionSemihardTripletSelector():
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, plotter, return_threshold=False, cpu=True, name='src'):
        self.cpu = cpu
        self.margin = margin
        self.return_threshold = return_threshold
        self.name = name

        self.plotter = plotter
        self.data_dict = utils.AverageData_Dict()

    def plot(self, epoch):

        self.plotter.plot('triplets number', 'epoch', 'num_triplet_{}'.format(self.name), 'Dual Triplets Mining',
                          epoch, self.data_dict['num_triplets'].last_avg())

        # Plot pairwise distances
        self.plotter.plot('distance', 'epoch', '{}_positive'.format(self.name), 'Pairwise mean distance',
                          epoch, self.data_dict['pos_center'].last_avg())
        self.plotter.plot('distance', 'epoch', '{}_negative'.format(self.name), 'Pairwise mean distance',
                          epoch, self.data_dict['neg_center'].last_avg())

    @torch.no_grad()
    def get_triplets(self, embeddings, labels, min_triplets=0):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = ml_utils.pdist(embeddings)
        distance_matrix = distance_matrix.cpu()
        np_distance_matrix = distance_matrix.cpu().detach().numpy()

        labels = labels.cpu().data.numpy()
        triplets = []
        easy_triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):

                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = semihard_negative(loss_values, self.margin)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
                else:
                    hard_negative = np.random.choice(negative_indices)
                    easy_triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        pos_center, neg_center = miner_utils.get_centers(np_distance_matrix, labels)
        self.data_dict['pos_center'].append(pos_center)
        self.data_dict['neg_center'].append(neg_center)
        self.data_dict['num_triplets'].append(len(triplets))

        if (min_triplets != 0) and (len(triplets) < min_triplets):
            num_missing = min_triplets - len(triplets)
            triplets += easy_triplets[:num_missing]
        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])
        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def solve(m1,m2,std1,std2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])

class FunctionPseudoTripletSelector():

    def __init__(self, margin, plotter, deadzone_ratio=0.1, cpu=False, name='tgt'):
        self.cpu = cpu
        self.margin = margin
        self.deadzone_ratio = deadzone_ratio
        self.name = name
        self.step = 0

        self.plotter = plotter
        self.data_dict = utils.AverageData_Dict()

    def plot(self, epoch):
        # Mining stats
        self.plotter.plot('triplets number', 'epoch', 'num_triplet_{}'.format(self.name), 'Dual Triplets Mining',
                     epoch, self.data_dict['num_triplets'].last_avg())

        # self.plotter.plot('margin', 'epoch', 'margin', 'Deadzone Margin',
        #              epoch, self.data_dict['deadzone_margin'].last_avg())

        # Plot pseudo labeling performannde
        self.plotter.plot('accuracy', 'epoch', 'pseudo-labeling', 'Pseudo-Labeling Accuracy',
                     epoch, self.data_dict['pseudo_labeling_accuracy'].last_avg())
        self.plotter.plot('accuracy', 'epoch', 'triplet_acc', 'Pseudo-Labeling Accuracy',
                          epoch, self.data_dict['triplet_selected_acc'].last_avg())
        self.plotter.plot('accuracy', 'epoch', 'triplet_tpr', 'Pseudo-Labeling Accuracy',
                     epoch, self.data_dict['triplet_selected_tpr'].last_avg())
        self.plotter.plot('accuracy', 'epoch', 'triplet_fpr', 'Pseudo-Labeling Accuracy',
                          epoch, self.data_dict['triplet_selected_fpr'].last_avg())

        # Plot pairwise distances
        self.plotter.plot('distance', 'epoch', '{}_positive'.format(self.name), 'Pairwise mean distance',
                          epoch, self.data_dict['pos_center'].last_avg())
        self.plotter.plot('distance', 'epoch', '{}_negative'.format(self.name), 'Pairwise mean distance',
                          epoch, self.data_dict['neg_center'].last_avg())

        self.plotter.plot('distance', 'epoch', 'id_thr', 'Pairwise mean distance',
                          epoch, self.data_dict['id_thr'].last_avg())
        self.plotter.plot('distance', 'epoch', 'pos_thr', 'Pairwise mean distance',
                          epoch, self.data_dict['pos_thr'].last_avg())
        self.plotter.plot('distance', 'epoch', 'neg_thr', 'Pairwise mean distance',
                          epoch, self.data_dict['neg_thr'].last_avg())

    @torch.no_grad()
    def get_triplets(self, embeddings, labels, _gmixture: GaussianMixture):
        if self.cpu:
            embeddings = embeddings.cpu()

        distance_matrix = ml_utils.pdist(embeddings)
        np_distance_matrix = distance_matrix.cpu().detach().numpy()

        ############################
        # Generate pseudo triplets #
        ############################

        # Extract values above the diagonal
        distances = np_distance_matrix[np.triu_indices(len(np_distance_matrix), k = 1)]

        # kmeans = KMeans(n_clusters=2, random_state=0).fit(np.expand_dims(distances, axis=1))
        #
        # pos_center = min(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[1][0])
        # neg_center = max(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[1][0])
        # id_thr = pos_center + neg_center / 2.0
        # self.data_dict['id_thr'].append(id_thr)

        # pos_center, neg_center = get_centers(np_distance_matrix, labels)
        # identification_threshold = (pos_center + neg_center) / 2.0
        # _gmixture.set_params(warm_start=True)

        gmixture = _gmixture.fit(np.expand_dims(distances, axis=1))

        # Visualisation debug
        #################################################
        # step = 0.05
        # max_distance = max(distances) + step
        # bins = np.arange(0.0, max_distance, step)
        # hist, _ = np.histogram(distances, bins=bins)
        #
        # self.plotter.stem_plot('gaussian_debug', 'Number of Samples', 'Distances Test {}'.format(self.step),
        #                       ['pos'], bins[1:], hist)
        #
        # Y = np.linspace(0, max(distances), 200)
        # funtion1 = gaussian(Y ,gmixture.means_[0][0], np.sqrt(gmixture.covariances_[0][0]))
        # funtion2 = gaussian(Y ,gmixture.means_[1][0], np.sqrt(gmixture.covariances_[1][0]))
        # X = np.column_stack((Y, Y))
        # Y = np.column_stack((funtion1, funtion2))
        # self.plotter.plot_curve('distance', 'epoch', ['g1', 'g2'], 'Gaussian update', X, Y)
        # self.step += 1
        #################################################

        id_thr = solve(gmixture.means_[0][0].item(),
                     gmixture.means_[1][0].item(),
                     np.sqrt(gmixture.covariances_[0][0].item()),
                     np.sqrt(gmixture.covariances_[1][0].item()))
        low_mean = min(gmixture.means_[0][0].item(), gmixture.means_[1][0].item())
        high_mean = max(gmixture.means_[0][0].item(), gmixture.means_[1][0].item())
        id_thr = id_thr[np.where(np.logical_and(id_thr>=low_mean, id_thr<=high_mean))][0]
        self.data_dict['id_thr'].append(id_thr)

        deadzone_margin = self.deadzone_ratio * self.margin
        positive_threshold = id_thr - deadzone_margin
        negative_threshold = id_thr + deadzone_margin
        self.data_dict['pos_thr'].append(positive_threshold)
        self.data_dict['neg_thr'].append(negative_threshold)

        triplets = []
        easy_triplet = [0,0,0]

        for target_idx in range(distance_matrix.shape[0]):
            label_indices = np.where(np_distance_matrix[target_idx] <= positive_threshold)[0][target_idx+1:]
            negative_indices = np.where(np_distance_matrix[target_idx] >= negative_threshold)[0][target_idx+1:]
            if (len(label_indices) < 2) or (len(negative_indices) == 0):
                continue

            anchor_positives = []
            for label_indice in label_indices:
                anchor_positives.append([target_idx, label_indice])

            if anchor_positives:
                anchor_positives = np.array(anchor_positives)
                ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
                for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                    negative_distance_matrix = distance_matrix[
                        torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)]

                    # loss = F.relu(distance_positive - distance_negative + self.margin)
                    loss_values = (ap_distance - negative_distance_matrix + self.margin)

                    loss_values = loss_values.data.cpu().numpy()

                    hard_negative = semihard_negative(loss_values, self.margin)
                    # hard_negative = np.random.choice(negative_indices)

                    if hard_negative is not None:
                        hard_negative = negative_indices[hard_negative]
                        triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
                    else:
                        hard_negative = np.random.choice(negative_indices)
                        easy_triplet = [anchor_positive[0], anchor_positive[1], hard_negative]

        pseudo_labeling_accuracy, _, _ = miner_utils.evaluate_pseudo_labeling(np_distance_matrix,
                                                                  labels,
                                                                  positive_threshold,
                                                                  negative_threshold)
        if triplets:
            triplet_selected_acc, triplet_selected_tpr, triplet_selected_fpr = miner_utils.evaluate_triplet(triplets,
                                                                                                            labels)
            self.data_dict['triplet_selected_acc'].append(triplet_selected_acc)
            self.data_dict['triplet_selected_tpr'].append(triplet_selected_tpr)
            self.data_dict['triplet_selected_fpr'].append(triplet_selected_fpr)
        self.data_dict['pseudo_labeling_accuracy'].append(pseudo_labeling_accuracy)

        pos_center, neg_center = miner_utils.get_centers(np_distance_matrix, labels)
        self.data_dict['pos_center'].append(pos_center)
        self.data_dict['neg_center'].append(neg_center)

        # Give at least one sample
        if len(triplets) == 0:
            triplets.append(easy_triplet)

        triplets = np.array(triplets)

        # Collect data
        self.data_dict['num_triplets'].append(len(triplets))
        # self.data_dict['deadzone_margin'].append(self.deadzone_ratio)

        return torch.LongTensor(triplets)


class FunctionDualTripletSelector3(TripletSelector):

    def __init__(self, margin, plotter, deadzone_ratio = 0.1, cpu=False):
        super(FunctionDualTripletSelector3, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.source_selector = FunctionSemihardTripletSelector(margin, plotter)
        self.target_selector = FunctionPseudoTripletSelector(margin, plotter, deadzone_ratio)

        self.gmixture = None

        self.plotter = plotter
        self.data_dict = utils.AverageData_Dict()

    def plot(self, epoch):
        # Mining stats
        self.plotter.plot('dualtriplet number', 'epoch', 'num dualtriplet', 'Dual Triplets Mining',
                     epoch, self.data_dict['num_dualtriplets'].last_avg())

        self.source_selector.plot(epoch)
        self.target_selector.plot(epoch)

    @torch.no_grad()
    def get_dualtriplet(self, source_embeddings, source_labels, target_embeddings, target_labels):

        # identification_threshold = miner_utils.get_threshold(target_embeddings, target_labels)

        # issame_matrix = np.zeros((len(source_labels), len(source_labels)), dtype=np.int)
        # for x in range(len(source_labels)):
        #     for y in range(len(source_labels)):
        #         if source_labels[x] == source_labels[y]:
        #             issame_matrix[x][y] = 1
        #
        # distance_matrix = ml_utils.pdist(source_embeddings)
        # np_distance_matrix = distance_matrix.cpu().detach().numpy()
        # distances = np_distance_matrix[np.triu_indices(len(np_distance_matrix), k=1)]
        # issame = issame_matrix[np.triu_indices(len(np_distance_matrix), k=1)]
        # gmixture = GaussianMixture(n_components=2).fit(np.expand_dims(distances, axis=1),
        #                                                y=np.expand_dims(issame, axis=1))

        # Visualisation debug
        ################################################
        # step = 0.05
        # max_distance = max(distances) + step
        # bins = np.arange(0.0, max_distance, step)
        # hist, _ = np.histogram(distances, bins=bins)
        #
        # self.plotter.stem_plot('gaussian_debug', 'Number of Samples', 'Distances Test',
        #                        ['pos'], bins[1:], hist)
        #################################################

        target_triplets = self.target_selector.get_triplets(target_embeddings, target_labels, self.gmixture)
        source_triplets = self.source_selector.get_triplets(source_embeddings, source_labels, min_triplets=len(target_triplets))

        num_dualtriplet = min(source_triplets.size(0), target_triplets.size(0))

        dual_triplets = torch.cat((source_triplets[:num_dualtriplet], target_triplets[:num_dualtriplet]), 1)

        # Collect data
        self.data_dict['num_dualtriplets'].append(dual_triplets.shape[0])

        return dual_triplets

class FunctionDualTripletSupervisedSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    Target samples are chosen ramdomly
    """

    def __init__(self, margin, plotter, cpu=False):
        super(FunctionDualTripletSupervisedSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.selector = FunctionSemihardTripletSelector(margin, plotter)

        self.plotter = plotter
        self.data_dict = utils.AverageData_Dict()

    def plot(self, epoch):
        # Mining stats
        self.plotter.plot('dualtriplet number', 'epoch', 'num dualtriplet', 'Dual Triplets Mining',
                     epoch, self.data_dict['num_dualtriplets'].last_avg())
        self.plotter.plot('dualtriplet number', 'epoch', 'num srctriplet', 'Dual Triplets Mining',
                     epoch, self.data_dict['num_srctriplets'].last_avg())
        self.plotter.plot('dualtriplet number', 'epoch', 'num tgttriplet', 'Dual Triplets Mining',
                     epoch, self.data_dict['num_tgttriplets'].last_avg())

    @torch.no_grad()
    def get_dualtriplet(self, source_embeddings, source_labels, target_embeddings, target_labels):

        target_triplets = self.selector.get_triplets(target_embeddings, target_labels)
        source_triplets = self.selector.get_triplets(source_embeddings, source_labels, min_triplets=len(target_triplets))
        self.data_dict['num_srctriplets'].append(len(source_triplets))
        self.data_dict['num_tgttriplets'].append(len(target_triplets))

        num_dualtriplet = min(source_triplets.size(0), target_triplets.size(0))

        dual_triplets = torch.cat((source_triplets[:num_dualtriplet], target_triplets[:num_dualtriplet]), 1)

        # Collect data
        self.data_dict['num_dualtriplets'].append(dual_triplets.shape[0])

        return dual_triplets
