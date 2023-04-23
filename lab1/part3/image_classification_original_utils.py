import sys
sys.path.append('../part12')
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from cv23_lab1_part3_utils import featuresSURF
from cv23_lab1_part3_utils import featuresHOG
from corner_detection_utils import CornerDetection
from corner_detection_utils import HarrisLaplacian
from blob_detection_utils import BlobDetection
from blob_detection_utils import HessianLaplacian
from box_detection_utils import BoxLaplacian


# define descriptors and detectors
# also define their names in order to
# accomodate easier file creation
descriptors = [lambda I, kp: featuresSURF(I, kp), lambda I, kp: featuresHOG(I, kp)]
detectors = [lambda i: CornerDetection(i, 2, 2.5, 0.005, 0.05),
            lambda i: HarrisLaplacian(i, 2, 2.5, 0.005, 0.05, 1.1, 6),
            lambda i: BlobDetection(i, 2, 0.005),
            lambda i: HessianLaplacian(i, 2, 0.005, 1.1, 6),
            lambda i: BoxLaplacian(i, 2, 0.005, 1.1, 6)]
descriptor_names = ["featuresSURF", "featuresHOG"]
detector_names = ["CornerDetection", "HarrisLaplacian", "BlobDetection", "HessianLaplacian", "BoxLaplacian"]
detector_descriptor = [(detector, descriptor)
                    for detector in detectors
                    for descriptor in descriptors]
detector_descriptor_names = [(detector, descriptor)
                            for detector in detector_names
                            for descriptor in descriptor_names]

# THIS FUNCTION IS DEFINED FOR EDUCATIONAL PURPOSES
# IT IS NOT USED BECAUSE IT IS EXTREMELY SLOW
def kmeans(X, k, max_iters=100):
    # X is the input data
    # k is the number of clusters to form
    # max_iters is the maximum number of iterations allowed
    # returns the centroids and their labels
    
    n_samples, _ = X.shape
    
    # Initialize the cluster centroids randomly
    centroids = X[np.random.choice(n_samples, size=k, replace=False)]
    
    for _ in range(max_iters):
        # Assign each data point to the nearest cluster centroid
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=-1) # shape (n_samples, k)
        labels = np.argmin(distances, axis=-1) # shape (n_samples,)
        
        # Update the cluster centroids as the mean of the data points in each cluster
        for i in range(k):
            mask = (labels == i)
            if np.sum(mask) > 0:
                centroids[i] = np.mean(X[mask], axis=0)
    
    return centroids, labels


def myFastBagOfVisualWords(train, test, clusters=500, maxiter=100):
    train_feature = np.concatenate(train, axis=0)
    #test_feature = np.concatenate(test, axis=0)

    # Set the number of clusters (visual words)
    # you can choose random number of clusters
    # clusters = random.randint(500, 2000)
    # clusters = 1000  # You can change this value

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=clusters, random_state=0, max_iter=maxiter, tol=0.001).fit(train_feature)

    # Function to compute the BoVW representation for a set of images
    def compute_bovw(feature_set):
        bovw_set = []
        for features in feature_set:
            distances = cdist(features, kmeans.cluster_centers_, metric='euclidean')
            labels = np.argmin(distances, axis=-1)
            counts = np.bincount(labels, minlength=clusters)
            l2norm = np.sqrt(np.sum(counts * counts))
            bovw = counts / l2norm
            bovw_set.append(bovw)
        return np.array(bovw_set)

    # Compute the BoVW representation for the training and testing sets
    train_bovw = compute_bovw(train)
    test_bovw = compute_bovw(test)

    return train_bovw, test_bovw
