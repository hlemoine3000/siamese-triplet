
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import tqdm
# import dlib
import numpy as np

class Clustering():

    def __init__(self, config):

        self.cluster_technic = None

        if config['Usage']['clustering_technic'] == 'dbscan':
            eps = config.getfloat('dbscan', 'eps')
            min_samples = config.getint('dbscan', 'min_samples')
            metric = config.get('dbscan', 'metric')
            self.cluster_technic = dbscan_cluster(eps,
                                                  min_samples,
                                                  metric)
        # elif config['Usage']['clustering_technic'] == 'chinese_whispers':
        #     threshold = config.getfloat('chinese_whispers', 'threshold')
        #     self.cluster_technic = chinese_whispers_cluster(threshold)
        else:
            raise ValueError('Clustering technique {} not supported. Check the configuration file.'.format(config['Usage']['clustering_technic']))

    def cluster(self, feature_array):
        # I have not define a standard output prediction yet
        # (dets_boxes, dets_scores, dets_classes, elapsed_time)
        # dets_boxes == [xmin, ymin, xmax, ymax]
        return self.cluster_technic.cluster(feature_array)


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
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        labels.append(cluster_labels)
        n_clusters.append(n_clusters_)
        eps_list.append(eps)

    best_n_clusters_idx = silhouette_scores.index(max(silhouette_scores))

    data_dict = {'silhouette_score': silhouette_scores[best_n_clusters_idx],
                 'n_cluster': n_clusters[best_n_clusters_idx],
                 'eps': eps_list[best_n_clusters_idx]}

    return labels[best_n_clusters_idx], data_dict


def kmeans_silhouetteanalysis(X, max_clusters, return_all=False, verbose=False):

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
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    if return_all:
        return labels, silhouette_scores, range_n_clusters
    else:
        best_n_clusters_idx = silhouette_scores.index(max(silhouette_scores))
        return labels[best_n_clusters_idx], silhouette_scores[best_n_clusters_idx], range_n_clusters[best_n_clusters_idx]


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
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # best_n_clusters_idx = silhouette_scores.index(max(silhouette_scores))

    return labels, silhouette_scores, range_n_clusters


class kmeans_cluster():
    def __init__(self,
                 n_clusters=2,
                 random_state=0):

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_technic = KMeans(n_clusters=n_clusters, random_state=random_state)

    def cluster(self, feature_array):

        kmeans = self.cluster_technic.fit(feature_array)
        return kmeans.labels_


class dbscan_cluster():
    def __init__(self,
                 eps,
                 min_samples,
                 metric):

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_technic = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)

    def cluster(self, feature_array):

        # Compute DBSCAN
        db = self.cluster_technic.fit(feature_array)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('')
        print('Estimated number of clusters: {}'.format(n_clusters_))
        print('Estimated number of noise points: {}'.format(n_noise_))

        return labels

class hac_cluster():

    def __init__(self,
                 n_clusters,
                 affinity='euclidean',
                 linkage='ward'):

        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage

        self.cluster_technic = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)

    def cluster(self, feature_array):

        self.cluster_technic.fit_predict(feature_array)
        return self.cluster_technic.labels_

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
