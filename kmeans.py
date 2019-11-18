from sklearn import datasets
import numpy as np

points, label = datasets.make_blobs(100, 2, centers=3)

def kmeans(points, k=3, threshold=0.01):
    # init centroids
    # centroids = np.random.randn(k, 2) # can use random number
    centroids = np.random.choice(points, k) # easier way is to randomly pick from k points
    # take the 1st point's vector length as initial error, can also use pair-wise distance's sum
    error = np.sum([x**2 + y**2 for x, y in points[0]]) 
    i = 0
    while(error > threshold):
        cluster_labels = assign_clusters(points, centroids)
        old_centroids, centroids = centroids, update_centroids(points, cluster_labels, k)
        error = np.sum([dist(oc, c) for oc, c in zip(old_centroids, centroids)])
        if i % 100==0:
            print(error)
    
    return cluster_labels, centroids

def dist(point, centroid):
    return np.sum([(point[i]-centroid[i])**2 for i in range(len(point))])

def get_label(point, centroids):
    label = np.argmin([dist(point, c) for c in centroids]) # optimized: direct compare during runtime
    return label

def assign_clusters(points, centroids):
    cluster_labels = [get_label(p, centroids) for p in points]
    return cluster_labels

def update_centroids(points, cluster_labels, k):
    clusters = [[] for i in range(k)]
    for i, label in enumerate(cluster_labels):
        clusters[label].append(points[i])
    centroids = [np.mean(cluster, axis=0) for cluster in clusters]
    return centroids

cluster_labels, centroids = kmeans(points, k=3)
