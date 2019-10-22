
import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def reduce_data(features: np.array,
                max_dots: int = 500) -> (np.array, np.array):

    features_to_reduce = features.copy()
    if features_to_reduce.shape[0] > max_dots:
        print('Too much samples. Selecting {} random samples.'.format(max_dots))
        chosen_indexes = random.sample(range(max_dots), max_dots)
        features_to_reduce = features_to_reduce[chosen_indexes]
    else:
        chosen_indexes = range(features.shape[0])

    return features_to_reduce, chosen_indexes

def tsne_projection(features: np.array,
                    max_dots: int = 500) -> (np.array, np.array):

    reduced_features, chosen_indexes = reduce_data(features, max_dots=max_dots)

    tsne_features = TSNE(n_components=2,
                         perplexity=30.0,
                         early_exaggeration=12.0, learning_rate=200.0, n_iter=20000,
                         n_iter_without_progress=1000, min_grad_norm=1e-7,
                         metric="euclidean", init="random", verbose=0,
                         random_state=None, method='exact', angle=0.2).fit_transform(reduced_features)

    return tsne_features, chosen_indexes

def pca_projection(features: np.array,
                   max_dots: int = 2000) -> (np.array, np.array):

    reduced_features, chosen_indexes = reduce_data(features, max_dots=max_dots)

    pca_features = PCA(n_components=2).fit_transform(reduced_features)

    return pca_features, chosen_indexes