
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tqdm
# import dlib
import numpy as np
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform

from utils import plotter


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def evaluate_clustering(features: np.array,
                        ground_truth: np.array,
                        cluster_techniques_list: list,
                        plotter: plotter.VisdomPlotter= None,
                        epoch:int=0,
                        max_clusters: int=10):

    if plotter:
        # Embeddings visualisation
        tsne_features = TSNE(n_components=2,
                             perplexity=30.0,
                             early_exaggeration=12.0, learning_rate=200.0, n_iter=20000,
                             n_iter_without_progress=1000, min_grad_norm=1e-7,
                             metric="euclidean", init="random", verbose=1,
                             random_state=None, method='exact', angle=0.2).fit_transform(features)
        plotter.scatter_plot('Embeddings Ground Truth',
                             tsne_features,
                             ground_truth)

    data_dict = {}
    for cluster_method in cluster_techniques_list:
        predictions, method_datadict = cluster_techniques(features,
                                                     cluster_method,
                                                     max_clusters=max_clusters)

        purity = purity_score(ground_truth, predictions)
        v_score = metrics.v_measure_score(ground_truth, predictions)

        print('Clustering {}'.format(cluster_method))
        print('Purity: {}'.format(purity))
        print('V measure score: {}'.format(v_score))

        if plotter:
            plotter.plot('metrics value', 'epoch', '{}_purity'.format(cluster_method), 'Clustering Performance', epoch,
                         np.mean(purity))
            plotter.plot('metrics value', 'epoch', '{}_v_score'.format(cluster_method), 'Clustering Performance', epoch,
                         np.mean(v_score))
            plotter.scatter_plot('{} Predictions'.format(cluster_method),
                                 tsne_features,
                                 predictions)

def cluster_techniques(features,
                       cluster_methods: str,
                       max_clusters: int = 10):

    if cluster_methods == 'kmeans':
        labels, data_dict = kmeans_silhouetteanalysis(features,
                                                      10,
                                                      verbose=False)
    elif cluster_methods == 'hac':
        labels, data_dict = hac_silhouetteanalysis(features,
                                                      10,
                                                      verbose=False)
    elif cluster_methods == 'dbscan':
        labels, data_dict = dbscan_silhouetteanalysis(features,
                                                      verbose=False)
    elif cluster_methods == 'spectral':
        labels, data_dict = spectral_analysis(features,
                                              10,
                                              verbose=False)
    else:
        raise ValueError('Clustering technique {} not supported.'.format(
            cluster_methods))

    return labels, data_dict


def dbscan_silhouetteanalysis(X, metric='euclidean', min_samples=1, verbose=False):

    #Reduce dimension
    features = X
    # pca = PCA(n_components=128, svd_solver='full')
    # pca.fit(X)
    # features = pca.transform(X)
    # print('Initial features shape: {}'.format(X.shape))
    # print('PCA features shape: {}'.format(features.shape))

    range_eps = np.arange(0.1, 1.0, 0.05)
    eps_list = []
    silhouette_scores = []
    labels = []
    n_clusters = []

    if verbose:
        print('Testing clustering.')
        tbar = tqdm.tqdm(range_eps)
    else:
        tbar = range_eps

    for eps in tbar:

        cluster_labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(features).labels_

        n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        # n_noise_ = list(cluster_labels).count(-1)


        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = metrics.silhouette_score(features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        labels.append(cluster_labels)
        n_clusters.append(n_clusters_)
        eps_list.append(eps)

    best_n_clusters_idx = silhouette_scores.index(max(silhouette_scores))

    data_dict = {'silhouette_score': silhouette_scores[best_n_clusters_idx],
                 'n_cluster': n_clusters[best_n_clusters_idx],
                 'eps': eps_list[best_n_clusters_idx]}

    return labels[best_n_clusters_idx], data_dict


def kmeans_silhouetteanalysis(X, max_clusters, verbose=False):

    range_n_clusters = range(2, max_clusters + 1)
    silhouette_scores = []
    labels = []

    if verbose:
        tbar = tqdm.tqdm(range_n_clusters)
    else:
        tbar = range_n_clusters
    for n_clusters in tbar:

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        labels.append(cluster_labels)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = metrics.silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    best_n_clusters_idx = silhouette_scores.index(max(silhouette_scores))
    data_dict = {'silhouette_score': silhouette_scores[best_n_clusters_idx],
                 'n_cluster': range_n_clusters[best_n_clusters_idx]}

    return labels[best_n_clusters_idx], data_dict


def hac_silhouetteanalysis(X,
                           max_clusters,
                           affinity='euclidean',
                           linkage='ward',
                           verbose=False):

    range_n_clusters = range(2, max_clusters + 1)
    silhouette_scores = []
    labels = []

    if verbose:
        tbar = tqdm.tqdm(range_n_clusters)
    else:
        tbar = range_n_clusters
    for n_clusters in tbar:

        # Initialize the clusterer with n_clusters value and a random generator
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
        cluster_labels = clusterer.fit_predict(X)
        labels.append(cluster_labels)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = metrics.silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    best_n_clusters_idx = silhouette_scores.index(max(silhouette_scores))
    data_dict = {'silhouette_score': silhouette_scores[best_n_clusters_idx],
                 'n_cluster': range_n_clusters[best_n_clusters_idx]}

    return labels[best_n_clusters_idx], data_dict


def spectral_analysis(X,
                      max_clusters,
                      verbose=False,
                      plotter: plotter.VisdomPlotter= None):

    affinity_matrix = getAffinityMatrix(X, k=7)
    nb_clusters, eigenvalues, eigenvectors = eigenDecomposition(affinity_matrix)

    clustering_thecnique = SpectralClustering(n_clusters=nb_clusters[0],
                                              assign_labels="discretize",
                                              random_state=0).fit(X)

    data_dict = {'eigendecomposition': nb_clusters,
                 'n_cluster': nb_clusters[0]}

    return clustering_thecnique.labels_, data_dict

def eigenDecomposition(A, plot=False, topK=5):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]

    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
    # the euclidean norm of complex numbers.
    #     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = np.linalg.eig(L)

    # if plot:
    #     plt.title('Largest eigen values of input matrix')
    #     plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
    #     plt.grid()

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1

    return nb_clusters, eigenvalues, eigenvectors


def getAffinityMatrix(coordinates, k=7):
    """
    Calculate affinity matrix based on input coordinates matrix and the numeber
    of nearest neighbours.

    Apply local scaling based on the k nearest neighbour
        References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    # calculate euclidian distance matrix
    dists = squareform(pdist(coordinates))

    # for each row, sort the distances ascendingly and take the index of the
    # k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T

    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix

def spectral_clustering(X, max_clusters, verbose=False):

    affinity_matrix = getAffinityMatrix(X, k=7)
    nb_clusters, eigenvalues, eigenvectors = eigenDecomposition(affinity_matrix)
    nb_clusters = min(nb_clusters[0], max_clusters)

# class chinese_whispers_cluster():
#     def __init__(self,
#                  threshold):
#
#         self.threshold = threshold
#
#     def cluster(self, feature_array):
#
#         feature_vectors = []
#         for feature in feature_array:
#             feature_vectors.append(dlib.vector(feature.tolist()))
#
#         labels = dlib.chinese_whispers_clustering(feature_vectors, self.threshold)
#
#         n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#         n_noise_ = list(labels).count(-1)
#         print('')
#         print('Estimated number of clusters: %d' % n_clusters_)
#         print('Estimated number of noise points: %d' % n_noise_)
#
#         return np.array(labels)
