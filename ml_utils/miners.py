from itertools import combinations
from sklearn.cluster import KMeans
import numpy as np
import torch

import utils
from ml_utils import clustering


def get_miner(mode, margin, people_per_batch, plotter, deadzone_ratio=0.1):

    if mode == 'dualtriplet':
        return FunctionDualTripletSelector2(margin,
                                            people_per_batch,
                                            plotter,
                                            deadzone_ratio=deadzone_ratio)

    elif mode == 'supervised_dualtriplet':
        return FunctionDualTripletSupervisedSelector(margin, plotter)

    else:
        raise Exception('Miner type {} does not exist.'.format(mode))


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


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
        distance_matrix = pdist(embeddings)

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

    def __init__(self, margin, plotter, return_threshold=False, cpu=True):
        self.cpu = cpu
        self.margin = margin
        self.return_threshold = return_threshold

        self.plotter = plotter
        self.data_dict = utils.AverageData_Dict()

    def plot(self, epoch):
        pass

    @torch.no_grad()
    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []
        positive_distances = []
        negative_distances = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            positive_distances.append(ap_distances)
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):

                negative_distances.append(distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)])

                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = semihard_negative(loss_values, self.margin)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        if self.return_threshold:
            positive_meandistance = torch.stack(positive_distances).mean().item()
            negative_meandistance = torch.stack(negative_distances).mean().item()
            identification_threshold = (positive_meandistance + negative_meandistance) / 2.0

            return torch.LongTensor(triplets), identification_threshold
        else:
            return torch.LongTensor(triplets)


class FunctionPseudoTripletSelector():
    def __init__(self, margin, plotter, cpu=False):
        self.cpu = cpu
        self.margin = margin
        self.deadzone_ratio = 0.1

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
        self.plotter.plot('distance', 'epoch', 'positive center', 'Cluster Centers',
                     epoch, self.data_dict['pos_center'].last_avg())
        self.plotter.plot('distance', 'epoch', 'negative center', 'Cluster Centers',
                     epoch, self.data_dict['neg_center'].last_avg())
        self.plotter.plot('margin', 'epoch', 'margin', 'Deadzone Margin',
                     epoch, self.data_dict['deadzone_margin'].last_avg())
        self.plotter.plot('accuracy', 'epoch', 'triplet selected', 'Pseudo-Labeling Accuracy',
                     epoch, self.data_dict['selectedtriplet_accuracy'].last_avg())
        self.plotter.plot('accuracy', 'epoch', 'pseudo-labeling', 'Pseudo-Labeling Accuracy',
                     epoch, self.data_dict['pseudolabeling_accuracy'].last_avg())

    @torch.no_grad()
    def get_triplets(self, embeddings, labels, identification_treshold):

        distance_matrix = pdist(embeddings)
        np_distance_matrix = distance_matrix.cpu().detach().numpy()
        # flatten_tgt_distance_matrix = np_tgt_distance_matrix.flatten()
        #
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(np.expand_dims(flatten_tgt_distance_matrix, axis=1))
        #
        # pos_center = min(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[1][0])
        # neg_center = max(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[1][0])
        # identification_treshold = pos_center + neg_center / 2.0

        triplets = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        deadzone_margin = self.deadzone_ratio * self.margin
        positive_threshold = distance_matrix - deadzone_margin
        negative_threshold = distance_matrix + deadzone_margin

        for target_idx in range(distance_matrix.shape[0]):
            label_indices = np.where(np_distance_matrix[target_idx] <= positive_threshold)[0]
            for idx in label_indices:
                if labels[target_idx] == labels[idx]:
                    tp += 1
                else:
                    fp += 1

            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np_distance_matrix[target_idx] >= negative_threshold)[0]
            for idx in negative_indices:
                if labels[target_idx] != labels[idx]:
                    tn += 1
                else:
                    fn += 1

            anchor_positives = []
            for label_indice in label_indices:
                if label_indice > target_idx:
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

                    # hard_negative = semihard_negative(loss_values, self.margin)
                    hard_negative = np.random.choice(negative_indices)

                    if hard_negative is not None:
                        # hard_negative = negative_indices[hard_negative]
                        triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        selectedtriplet_accuracy = (tp + tn) / (tp + tn + fp + fn)

        tn = 0
        tp = 0
        fp = 0
        fn = 0
        for triplet in triplets:
            anchor_idx = triplet[0]
            pos_idx = triplet[1]
            neg_idx = triplet[2]

            if labels[anchor_idx] == labels[pos_idx]:
                tp += 1
            else:
                fp += 1
            if labels[anchor_idx] != labels[neg_idx]:
                tn += 1
            else:
                fn += 1
        if (tp + tn + fp + fn) != 0:
            pseudolabeling_accuracy = (tp + tn) / (tp + tn + fp + fn)
        else:
            pseudolabeling_accuracy = 0.0
        # print('Target triplet accuracy: {}'.format(target_triplet_label_accuracy))

        # Give at least one sample
        if len(triplets) == 0:
            triplets.append([0, 0, 0, 0, 0, 0])
        triplets = np.array(triplets)

        # Collect data
        self.data_dict['num_tgttriplets'].append(len(triplets))
        self.data_dict['deadzone_margin'].append(self.deadzone_ratio)
        self.data_dict['pseudolabeling_accuracy'].append(pseudolabeling_accuracy)
        self.data_dict['selectedtriplet_accuracy'].append(selectedtriplet_accuracy)

        return torch.LongTensor(triplets)

class FunctionDualTripletSelector2():
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    Target samples are chosen ramdomly
    """

    def __init__(self, margin, max_clusters, plotter, deadzone_ratio=0.1, cpu=False):
        self.cpu = cpu
        self.margin = margin
        self.max_clusters = max_clusters
        self.deadzone_ratio = deadzone_ratio

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

        self.plotter.plot('distance', 'epoch', 'positive center', 'Cluster Centers',
                     epoch, self.data_dict['pos_center'].last_avg())
        self.plotter.plot('distance', 'epoch', 'negative center', 'Cluster Centers',
                     epoch, self.data_dict['neg_center'].last_avg())
        self.plotter.plot('margin', 'epoch', 'margin', 'Deadzone Margin',
                     epoch, self.data_dict['deadzone_margin'].last_avg())

        self.plotter.plot('accuracy', 'epoch', 'triplet accuracy', 'Pseudo-Labeling Accuracy',
                     epoch, self.data_dict['target_triplet_label_accuracy'].last_avg())
        self.plotter.plot('accuracy', 'epoch', 'triplet specificity', 'Pseudo-Labeling Accuracy',
                          epoch, self.data_dict['target_triplet_label_specificity'].last_avg())
        self.plotter.plot('accuracy', 'epoch', 'pseudo-labeling', 'Pseudo-Labeling Accuracy',
                     epoch, self.data_dict['target_labeling_accuracy'].last_avg())

        self.plotter.plot('distance', 'epoch', 'id_thr', 'Pairwise mean distance',
                     epoch, self.data_dict['identification_threshold'].last_avg())
        self.plotter.plot('distance', 'epoch', 'pos_thr', 'Pairwise mean distance',
                     epoch, self.data_dict['positive_threshold'].last_avg())
        self.plotter.plot('distance', 'epoch', 'neg_thr', 'Pairwise mean distance',
                     epoch, self.data_dict['negative_threshold'].last_avg())

    @torch.no_grad()
    def get_dualtriplet(self, source_embeddings, source_labels, target_embeddings, target_labels):
        if self.cpu:
            source_embeddings = source_embeddings.cpu()
            target_embeddings = target_embeddings.cpu()

        concat_embeddings = torch.cat((source_embeddings, target_embeddings), 0)
        distance_matrix = pdist(concat_embeddings)
        dist_split_idx = source_labels.size(0)

        ############################
        # Generate source triplets #
        ############################

        src_distance_matrix = distance_matrix[:dist_split_idx, :dist_split_idx]
        positive_distances = []
        negative_distances = []

        source_triplets = []
        for label in set(source_labels):
            label_mask = (source_labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = src_distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            positive_distances.append(ap_distances)
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):

                negative_distance_matrix = src_distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)]
                negative_distances.append(negative_distance_matrix)

                # loss = F.relu(distance_positive - distance_negative + self.margin)
                loss_values = (ap_distance - negative_distance_matrix + self.margin)

                loss_values = loss_values.data.cpu().numpy()

                # hard_negative = np.random.choice(negative_indices)
                hard_negative = semihard_negative(loss_values, self.margin)

                if hard_negative is not None:

                    hard_negative = negative_indices[hard_negative]
                    source_triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        source_positive_meandistance = torch.stack(positive_distances).mean().item()
        source_negative_meandistance = torch.stack(negative_distances).mean().item()
        identification_threshold = (source_positive_meandistance + source_negative_meandistance) / 2.0

        self.data_dict['pos_center'].append(source_positive_meandistance)
        self.data_dict['neg_center'].append(source_negative_meandistance)

        ############################
        # Generate target triplets #
        ############################

        tgt_distance_matrix = distance_matrix[dist_split_idx:, dist_split_idx:]
        np_tgt_distance_matrix = tgt_distance_matrix.cpu().detach().numpy()
        # flatten_tgt_distance_matrix = np_tgt_distance_matrix.flatten()
        #
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(np.expand_dims(flatten_tgt_distance_matrix, axis=1))
        #
        # pos_center = min(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[1][0])
        # neg_center = max(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[1][0])
        # identification_treshold = pos_center + neg_center / 2.0

        deadzone_margin = self.deadzone_ratio * self.margin
        positive_threshold = identification_threshold - deadzone_margin
        negative_threshold = identification_threshold + deadzone_margin

        self.data_dict['identification_threshold'].append(identification_threshold)
        self.data_dict['positive_threshold'].append(positive_threshold)
        self.data_dict['negative_threshold'].append(negative_threshold)

        target_triplets = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for target_idx in range(tgt_distance_matrix.shape[0]):
            label_indices = np.where(np_tgt_distance_matrix[target_idx] <= positive_threshold)[0]

            for idx in label_indices:
                if target_labels[target_idx] == target_labels[idx]:
                    tp += 1
                else:
                    fp += 1

            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np_tgt_distance_matrix[target_idx] >= negative_threshold)[0]
            for idx in negative_indices:
                if target_labels[target_idx] != target_labels[idx]:
                    tn += 1
                else:
                    fn += 1

            anchor_positives = []
            for label_indice in label_indices:
                if label_indice > target_idx:
                    anchor_positives.append([target_idx, label_indice])

            if anchor_positives:
                anchor_positives = np.array(anchor_positives)
                ap_distances = tgt_distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
                for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                    negative_distance_matrix = tgt_distance_matrix[
                        torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)]

                    # loss = F.relu(distance_positive - distance_negative + self.margin)
                    loss_values = (ap_distance - negative_distance_matrix + self.margin)

                    loss_values = loss_values.data.cpu().numpy()

                    hard_negative = semihard_negative(loss_values, self.margin)
                    # hard_negative = np.random.choice(negative_indices)

                    if hard_negative is not None:
                        hard_negative = negative_indices[hard_negative]
                        target_triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        # print('\nDead zone: {}'.format(self.deadzone_ratio))
        target_labeling_accuracy = (tp + tn) / (tp + tn + fp + fn)
        # print('Labeling accuracy: {}'.format(target_labeling_accuracy))

        tn = 0
        tp = 0
        fp = 0
        fn = 0
        for triplet in target_triplets:
            anchor_idx = triplet[0]
            pos_idx = triplet[1]
            neg_idx = triplet[2]

            if target_labels[anchor_idx] == target_labels[pos_idx]:
                tp += 1
            else:
                fp += 1
            if target_labels[anchor_idx] != target_labels[neg_idx]:
                tn += 1
            else:
                fn += 1
        if (tp + tn + fp + fn) != 0:
            target_triplet_label_accuracy = (tp + tn) / (tp + tn + fp + fn)
        else:
            target_triplet_label_accuracy = 0.0

        if (tn + fp) != 0:
            target_triplet_label_specificity = fp / (fp + tn)
        else:
            target_triplet_label_specificity = 0.0
        # print('Target triplet accuracy: {}'.format(target_triplet_label_accuracy))

        # Fuse and balance triplets
        dual_triplets = []
        num_dualtriplets = min(len(source_triplets), len(target_triplets))

        for idx in range(num_dualtriplets):
            dual_triplets.append(source_triplets[idx] + target_triplets[idx])

        # Give at least one sample
        if len(dual_triplets) == 0:
            dual_triplets.append([0, 0, 0, 0, 0, 0])

        dual_triplets = np.array(dual_triplets)

        # Collect data
        self.data_dict['num_dualtriplets'].append(len(dual_triplets))
        self.data_dict['num_srctriplets'].append(len(source_triplets))
        self.data_dict['num_tgttriplets'].append(len(target_triplets))
        self.data_dict['deadzone_margin'].append(self.deadzone_ratio)

        self.data_dict['target_triplet_label_accuracy'].append(target_triplet_label_accuracy)
        self.data_dict['target_labeling_accuracy'].append(target_labeling_accuracy)
        self.data_dict['target_triplet_label_specificity'].append(target_triplet_label_specificity)

        return torch.LongTensor(dual_triplets)


# class FunctionDualTripletSelector(TripletSelector):
#     """
#     For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
#     Margin should match the margin used in triplet loss.
#     negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
#     and return a negative index for that pair
#     Target samples are chosen ramdomly
#     """
#
#     def __init__(self, margin, max_clusters, cpu=False):
#         super(FunctionDualTripletSelector, self).__init__()
#         self.cpu = cpu
#         self.margin = margin
#         self.max_clusters = max_clusters
#
#     def get_dualtriplet(self, source_embeddings, source_labels, target_embeddings, target_labels):
#         if self.cpu:
#             source_embeddings = source_embeddings.cpu()
#             target_embeddings = target_embeddings.cpu()
#
#         concat_embeddings = torch.cat((source_embeddings, target_embeddings), 0)
#         distance_matrix = pdist(concat_embeddings)
#         dist_split_idx = source_labels.size(0)
#
#         ############################
#         # Generate source triplets #
#         ############################
#
#         src_distance_matrix = distance_matrix[:dist_split_idx, :dist_split_idx]
#
#         source_triplets = []
#         for label in set(source_labels):
#             label_mask = (source_labels == label)
#             label_indices = np.where(label_mask)[0]
#             if len(label_indices) < 2:
#                 continue
#             negative_indices = np.where(np.logical_not(label_mask))[0]
#             anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
#             anchor_positives = np.array(anchor_positives)
#
#             ap_distances = src_distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
#             for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
#
#                 negative_distance_matrix = src_distance_matrix[
#                     torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)]
#
#                 # loss = F.relu(distance_positive - distance_negative + self.margin)
#                 loss_values = (ap_distance - negative_distance_matrix + self.margin)
#
#                 loss_values = loss_values.data.cpu().numpy()
#
#                 hard_negative = semihard_negative(loss_values, self.margin)
#
#                 if hard_negative is not None:
#
#                     hard_negative = negative_indices[hard_negative]
#                     source_triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
#
#         ############################
#         # Generate target triplets #
#         ############################
#
#         tgt_distance_matrix = distance_matrix[dist_split_idx:, dist_split_idx:]
#
#         # target_clustering = clustering.kmeans_cluster(n_clusters=15)
#         # target_labels = target_clustering.cluster(target_embeddings.cpu().detach().numpy())
#
#         target_labels, clustering_score, n_clusters = clustering.kmeans_silhouetteanalysis(target_embeddings.cpu().detach().numpy(),
#                                                                                            self.max_clusters)
#
#         target_triplets = []
#         for label in set(target_labels):
#             label_mask = (target_labels == label)
#             label_indices = np.where(label_mask)[0]
#             if len(label_indices) < 2:
#                 continue
#             negative_indices = np.where(np.logical_not(label_mask))[0]
#             anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
#             anchor_positives = np.array(anchor_positives)
#
#             ap_distances = tgt_distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
#             for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
#
#                 negative_distance_matrix = tgt_distance_matrix[
#                     torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)]
#
#                 # loss = F.relu(distance_positive - distance_negative + self.margin)
#                 loss_values = (ap_distance - negative_distance_matrix + self.margin)
#
#                 loss_values = loss_values.data.cpu().numpy()
#
#                 hard_negative = semihard_negative(loss_values, self.margin)
#
#                 if hard_negative is not None:
#                     hard_negative = negative_indices[hard_negative]
#                     target_triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
#
#         # Fuse and balance triplets
#
#         dual_triplets = []
#         num_dualtriplets = min(len(source_triplets), len(target_triplets))
#         for idx in range(num_dualtriplets):
#             dual_triplets.append(source_triplets[idx] + target_triplets[idx])
#
#         # Give at least one sample
#         if len(dual_triplets) == 0:
#             dual_triplets.append([0, 0, 0, 0, 0, 0])
#
#         dual_triplets = np.array(dual_triplets)
#
#         # miner_data = {'num_src_triplets': len(source_triplets),
#         #               'num_tgt_triplets': len(target_triplets)}
#
#         miner_data = {'num_src_triplets': len(source_triplets),
#                       'num_tgt_triplets': len(target_triplets),
#                       'n_clusters': n_clusters,
#                       'clustering_score': clustering_score}
#
#         return torch.LongTensor(dual_triplets), miner_data


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

        source_triplets = self.selector.get_triplets(source_embeddings, source_labels)
        target_triplets = self.selector.get_triplets(target_embeddings, target_labels)

        num_src_triplets = source_triplets.size(0)
        num_tgt_triplets = target_triplets.size(0)
        num_dualtriplet = min(num_src_triplets, num_tgt_triplets)

        source_triplets = source_triplets[:num_dualtriplet]
        target_triplets = target_triplets[:num_dualtriplet]

        dual_triplets = torch.cat((source_triplets, target_triplets), 1)

        # Collect data
        self.data_dict['num_dualtriplets'].append(dual_triplets.shape[0])
        self.data_dict['num_srctriplets'].append(len(source_triplets))
        self.data_dict['num_tgttriplets'].append(len(target_triplets))

        return dual_triplets


# def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
#                                                                                  negative_selection_fn=hardest_negative,
#                                                                                  cpu=cpu)
#
#
# def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
#                                                                                 negative_selection_fn=random_hard_negative,
#                                                                                 cpu=cpu)
#
#
# def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
#                                                                                   negative_selection_fn=lambda x: semihard_negative(x, margin),
#                                                                                   cpu=cpu)
#
#
# def SemihardNegativeQuadrupletSelector(margin, cpu=False): return FunctionNegativeQuadrupletSelector(margin=margin,
#                                                                                   negative_selection_fn=lambda x: semihard_negative(x, margin),
#                                                                                   cpu=cpu)


# def SemihardNegativeTargetQuadrupletSelector(margin, cpu=False): return FunctionNegativeTargetQuadrupletSelector(margin=margin,
#                                                                                   cpu=cpu)
