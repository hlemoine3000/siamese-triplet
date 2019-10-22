
import numpy as np
from ml_utils import ml_utils


def get_threshold(embeddings, labels):
    np_distance_matrix = ml_utils.pdist(embeddings).cpu().detach().numpy()
    pos_center, neg_center = get_centers(np_distance_matrix, labels)
    return pos_center + 1/3 * (pos_center + neg_center)
    # return (pos_center + neg_center) / 2.0


def get_centers(distance_matrix: np.array,
                ground_truth):

    pos_center_list = []
    neg_center_list = []
    for x in range(distance_matrix.shape[0]):
        for y in range(x+1, distance_matrix.shape[1]):
            if ground_truth[x] == ground_truth[y]:
                pos_center_list.append(distance_matrix[x,y])
            else:
                neg_center_list.append(distance_matrix[x,y])

    pos_center = sum(pos_center_list) / len(pos_center_list)
    neg_center = sum(neg_center_list) / len(neg_center_list)

    return pos_center, neg_center


def evaluate_pseudo_labeling(distance_matrix,
                             ground_truth,
                             pos_thr,
                             neg_thr):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for sample_idx in range(distance_matrix.shape[0]):
        positive_indices = np.where(distance_matrix[sample_idx] <= pos_thr)[0][sample_idx+1:]

        for idx in positive_indices:
            if ground_truth[sample_idx] == ground_truth[idx]:
                tp += 1
            else:
                fn += 1

        negative_indices = np.where(distance_matrix[sample_idx] >= neg_thr)[0][sample_idx+1:]
        for idx in negative_indices:
            if ground_truth[sample_idx] != ground_truth[idx]:
                tn += 1
            else:
                fp += 1

    acc = 0 if (tp + tn + fp + fn == 0) else float(tp + tn) / float(tp + tn + fp + fn)
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)

    return acc, tpr, fpr


def evaluate_triplet(triplets,
                     ground_truth):
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    for triplet in triplets:
        anchor_idx = triplet[0]
        pos_idx = triplet[1]
        neg_idx = triplet[2]

        if ground_truth[anchor_idx] == ground_truth[pos_idx]:
            tp += 1
        else:
            fn += 1
        if ground_truth[anchor_idx] != ground_truth[neg_idx]:
            tn += 1
        else:
            fp += 1

    acc = 0 if (tp + tn + fp + fn == 0) else float(tp + tn) / float(tp + tn + fp + fn)
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)

    return acc, tpr, fpr