import os
import random

from random import choice, shuffle
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

random.seed(12345)
train = pd.read_csv('input/sarogata.csv')
x_train = np.array(train.iloc[:, 1:], dtype='float32')

print(train)

def kmeans(x, n_features, n_clusters, n_max_steps=10000, early_stop=0.0):
    input_vec = tf.constant(x, dtype=tf.float32)
    centroids = tf.Variable(tf.slice(tf.random_shuffle(input_vec), [0, 0], [n_clusters, -1]), dtype=tf.float32)
    old_centroids = tf.Variable(tf.zeros([n_clusters, n_features]), dtype=tf.float32)
    centroid_distance = tf.Variable(tf.zeros([n_clusters, n_features]))
    expanded_vectors = tf.expand_dims(input_vec, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)

    assignments = tf.argmin(distances, 0)
    
    means = tf.concat([tf.reduce_mean(
        tf.gather(input_vec, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
        reduction_indices=[1]) for c in range(n_clusters)], 0)

    save_old_centroids = tf.assign(old_centroids, centroids)

    update_centroids = tf.assign(centroids, means)
    init_op = tf.global_variables_initializer()

    performance = tf.assign(centroid_distance, tf.subtract(centroids, old_centroids))
    check_stop = tf.reduce_sum(tf.abs(performance))
    calc_wss = tf.reduce_sum(tf.reduce_min(distances, 0))

    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(n_max_steps):
            sess.run(save_old_centroids)
            _, centroid_values, assignment_values = sess.run(
                [update_centroids, centroids, assignments])            
            sess.run(calc_wss)
            sess.run(check_stop)
            current_stop_coeficient = check_stop.eval()
            wss = calc_wss.eval()
            print(step, current_stop_coeficient)

            if current_stop_coeficient <= early_stop:
                break
    return centroid_values, assignment_values, wss

wcss_list = []

for i in range(2, 10):
    centers, cluster_assignments, wcss = kmeans(x_train, len(x_train[0]), i)
    wcss_list.append(wcss)

plt.figure(figsize=(12, 24))
plt.subplot(211)
plt.plot(range(2, 10), wcss_list)
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.title("WCSS vs Cluters")

# From the plot we can see the elbow is occuring when cluster number is 5
centers, cluster_assignments, wss = kmeans(x_train, len(x_train[0]), 5)
pca_model = PCA(n_components=3)
reduced_data = pca_model.fit_transform(x_train)
reduced_centers = pca_model.transform(centers)


plt.subplot(212, projection='3d')
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=cluster_assignments)
plt.title("Clusters")
plt.show()
