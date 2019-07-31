from itertools import combinations

from ml_utils import clustering
import numpy as np
import torch


def get_miner(mode, margin, people_per_batch):

    if mode == 'dualtriplet':
        return FunctionDualTripletSelector(margin,
                                           people_per_batch)

    elif mode == 'supervised_dualtriplet':
        return FunctionDualTripletSupervisedSelector(margin)

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


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

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
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


class FunctionNegativeQuadrupletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    Target samples are chosen ramdomly
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeQuadrupletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_quadruplets(self, source_embeddings, target_embeddings, source_labels):
        if self.cpu:
            source_embeddings = source_embeddings.cpu()
        distance_matrix = pdist(source_embeddings)
        distance_matrix = distance_matrix.cpu()

        num_target = target_embeddings.size(0)

        labels = source_labels.cpu().data.numpy()
        quadruplets = []

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
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:

                    hard_negative = negative_indices[hard_negative]

                    # Target sample chosen ramdomly
                    target_indice = np.random.randint(num_target)

                    quadruplets.append([anchor_positive[0], anchor_positive[1], hard_negative, target_indice])

        if len(quadruplets) == 0:

            # Target sample chosen ramdomly
            target_indice = np.random.randint(num_target)

            quadruplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0], target_indice])

        quadruplets = np.array(quadruplets)

        return torch.LongTensor(quadruplets)


class FunctionDualTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    Target samples are chosen ramdomly
    """

    def __init__(self, margin, max_clusters, cpu=False):
        super(FunctionDualTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.max_clusters = max_clusters

    def get_dualtriplet(self, source_embeddings, source_labels, target_embeddings):
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
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):

                negative_distance_matrix = src_distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)]

                # loss = F.relu(distance_positive - distance_negative + self.margin)
                loss_values = (ap_distance - negative_distance_matrix + self.margin)

                loss_values = loss_values.data.cpu().numpy()

                hard_negative = semihard_negative(loss_values, self.margin)

                if hard_negative is not None:

                    hard_negative = negative_indices[hard_negative]
                    source_triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        ############################
        # Generate target triplets #
        ############################

        tgt_distance_matrix = distance_matrix[dist_split_idx:, dist_split_idx:]

        # target_clustering = clustering.kmeans_cluster(n_clusters=15)
        # target_labels = target_clustering.cluster(target_embeddings.cpu().detach().numpy())

        target_labels, clustering_score, n_clusters = clustering.kmeans_silhouetteanalysis(target_embeddings.cpu().detach().numpy(),
                                                                                           self.max_clusters)

        target_triplets = []
        for label in set(target_labels):
            label_mask = (target_labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = tgt_distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):

                negative_distance_matrix = tgt_distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)]

                # loss = F.relu(distance_positive - distance_negative + self.margin)
                loss_values = (ap_distance - negative_distance_matrix + self.margin)

                loss_values = loss_values.data.cpu().numpy()

                hard_negative = semihard_negative(loss_values, self.margin)

                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    target_triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        # Fuse and balance triplets
        dual_triplets = []
        num_dualtriplets = min(len(source_triplets), len(target_triplets))
        for idx in range(num_dualtriplets):
            dual_triplets.append(source_triplets[idx] + target_triplets[idx])

        # Give at least one sample
        if len(dual_triplets) == 0:
            dual_triplets.append([0, 0, 0, 0, 0, 0])

        dual_triplets = np.array(dual_triplets)

        # miner_data = {'num_src_triplets': len(source_triplets),
        #               'num_tgt_triplets': len(target_triplets)}

        miner_data = {'num_src_triplets': len(source_triplets),
                      'num_tgt_triplets': len(target_triplets),
                      'n_clusters': n_clusters,
                      'clustering_score': clustering_score}

        return torch.LongTensor(dual_triplets), miner_data


class FunctionDualTripletSupervisedSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    Target samples are chosen ramdomly
    """

    def __init__(self, margin, cpu=False):
        super(FunctionDualTripletSupervisedSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.selector = SemihardNegativeTripletSelector(margin)

    def get_dualtriplet(self, source_embeddings, source_labels, target_embeddings, target_labels):

        source_triplets = self.selector.get_triplets(source_embeddings, source_labels)
        target_triplets = self.selector.get_triplets(target_embeddings, target_labels)

        num_src_triplets = source_triplets.size(0)
        num_tgt_triplets = target_triplets.size(0)
        num_dualtriplet = min(num_src_triplets, num_tgt_triplets)

        source_triplets = source_triplets[:num_dualtriplet]
        target_triplets = target_triplets[:num_dualtriplet]

        dual_triplets = torch.cat((source_triplets, target_triplets), 1)

        miner_data = {'num_src_triplets': num_src_triplets,
                      'num_tgt_triplets': num_tgt_triplets}

        return dual_triplets, miner_data


# class FunctionQuintupletSelector(TripletSelector):
#     """
#     For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
#     Margin should match the margin used in triplet loss.
#     negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
#     and return a negative index for that pair
#     Target samples are chosen ramdomly
#     """
#
#     def __init__(self, src_margin, tgt_margin, lamda, deadzone=0.1, cpu=False):
#         super(FunctionQuintupletSelector, self).__init__()
#         self.cpu = cpu
#         self.src_margin = src_margin
#         self.tgt_margin = tgt_margin
#         self.deadzone = deadzone
#         self.lamda = lamda
#
#     def get_quintuplet(self, source_embeddings, source_labels, target_embeddings):
#         if self.cpu:
#             source_embeddings = source_embeddings.cpu()
#             target_embeddings = target_embeddings.cpu()
#
#         concat_embeddings = torch.cat((source_embeddings, target_embeddings), 0)
#         distance_matrix = pdist(concat_embeddings)
#         dist_split_idx = source_labels.size(0)
#
#         num_target = target_embeddings.size(0)
#
#         ############################
#         # Generate source triplets #
#         ############################
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
#             ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
#             for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
#
#                 negative_distance_matrix = distance_matrix[
#                     torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)]
#
#                 # loss1 = F.relu(distance_positive - distance_negative + self.margin)
#                 loss_values = self.lamda[0] * (ap_distance - negative_distance_matrix + self.src_margin)
#
#                 loss_values = loss_values.data.cpu().numpy()
#
#                 hard_negative = semihard_negative(loss_values, self.src_margin)
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
#         # Calculate mean ap and an distances based on source domain
#         ap_distances_list = []
#         an_distances_list = []
#         for label in set(source_labels):
#             label_mask = (source_labels == label)
#             label_indices = np.where(label_mask)[0]
#             if len(label_indices) < 2:
#                 continue
#             negative_indices = np.where(np.logical_not(label_mask))[0]
#             anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
#             anchor_positives = np.array(anchor_positives)
#
#             ap_distances_list.append(
#                 torch.mean(distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]))
#
#             for anchor_positive in anchor_positives:
#                 an_distances_list.append(torch.mean(distance_matrix[anchor_positive[0], negative_indices]))
#
#         mean_ap_distances = torch.mean(torch.stack(ap_distances_list)).cpu().item()
#         mean_an_distances = torch.mean(torch.stack(an_distances_list)).cpu().item()
#         if mean_an_distances < mean_ap_distances:
#             mean_an_distances = torch.Tensor(2)
#             mean_ap_distances = torch.Tensor(1)
#
#         similarity_threshold = (mean_an_distances + mean_ap_distances) / 2
#         similar_threshold = similarity_threshold - self.deadzone
#         different_threshold = similarity_threshold + self.deadzone
#
#         # Form triplets based on the similarity distances
#         target_triplets = []
#         target_distances = distance_matrix[dist_split_idx:, dist_split_idx:].detach().cpu().numpy()
#         target_idxs = np.arange(num_target)
#         for target_idx in target_idxs:
#             target_distance = target_distances[target_idx, :]
#
#             # Search for positive samples
#             # Reject values over the similar_threshold and search for the highest distance.
#             positive_samples_idxs = np.where(target_distance < similar_threshold)
#             ap_distances = target_distance[positive_samples_idxs]
#             if ap_distances.size != 0:
#                 max_ap_distance = np.amax(ap_distances)
#                 positive_index = np.where(ap_distances == max_ap_distance)[0][0]
#             else:
#                 positive_index = None
#
#             # Search for negative samples
#             # Reject values below the different_threshold and search for the lowest distance.
#             negative_samples_idxs = np.where(target_distance > different_threshold)
#             an_distances = target_distance[negative_samples_idxs]
#             if an_distances.size != 0:
#                 min_an_distance = np.amin(an_distances)
#                 negative_index = np.where(an_distances == min_an_distance)[0][0]
#             else:
#                 negative_index = None
#
#             if (negative_index is not None) and (positive_index is not None):
#                 loss_value = target_distance[positive_index] - target_distance[negative_index] + self.tgt_margin
#                 if loss_value > 0:
#                     target_triplets.append([target_idx, positive_index, negative_index])
#
#         # Fuse and balance triplets
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
#         miner_data = {'num_src_triplets': len(source_triplets),
#                       'num_tgt_triplets': len(target_triplets)}
#
#         return torch.LongTensor(dual_triplets), miner_data


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)


def SemihardNegativeQuadrupletSelector(margin, cpu=False): return FunctionNegativeQuadrupletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)


# def SemihardNegativeTargetQuadrupletSelector(margin, cpu=False): return FunctionNegativeTargetQuadrupletSelector(margin=margin,
#                                                                                   cpu=cpu)
