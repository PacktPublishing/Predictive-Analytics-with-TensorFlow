import tensorflow as tf
import numpy as np
import pandas as pd
import time
import readers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from tensorflow.python.framework import ops
import warnings
import random
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

# Data Parameters
tf.flags.DEFINE_string("data_file", "Input/ratings.dat", "Data source for the positive data.")
tf.flags.DEFINE_string("K", 5, "Number of clusters")
tf.flags.DEFINE_string("MAX_ITERS", 1000, "Maximum number of iterations")
tf.flags.DEFINE_string("TRAINED", False, "Use TRAINED user vs item matrix")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count

def k_mean_clustering(ratings_df,K=5,MAX_ITERS = 1000,TRAINED=False):
    """
    Return movies alongwith there respective clusters
    INPUTS :
        ratings_df : rating dataframe, store all users rating for respective movies
        K          : number of clusters
        MAX_ITERS  : maximum number of recommendation
        TRAINED    : TRUE or FALSE, weather use trained user vs movie table or untrained
    OUTPUT:
        List of movies/items and list of clusters
    """
    if TRAINED:
        df=pd.read_pickle("user_item_table_train.pkl")
    else:
        df=pd.read_pickle("user_item_table.pkl")
    df = df.T

    start = time.time()
    N=df.shape[0]

    points = tf.Variable(df.as_matrix())
    cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))

    # Silly initialization:  Use the first K points as the starting
    # centroids.  In the real world, do this better.
    centroids = tf.Variable(tf.slice(points.initialized_value(), [0,0], [K,df.shape[1]]))

    # Replicate to N copies of each centroid and K copies of each
    # point, then subtract and compute the sum of squared distances.
    rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, df.shape[1]])
    rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, df.shape[1]])
    sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
                                reduction_indices=2)

    # Use argmin to select the lowest-distance point
    best_centroids = tf.argmin(sum_squares, 1)
    did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids, cluster_assignments))

    means = bucket_mean(points, best_centroids, K)

    # Do not write to the assigned clusters variable until after
    # computing whether the assignments have changed - hence with_dependencies
    with tf.control_dependencies([did_assignments_change]):
        do_updates = tf.group(
            centroids.assign(means),
            cluster_assignments.assign(best_centroids))

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    changed = True
    iters = 0

    while changed and iters < MAX_ITERS:
        iters += 1
        [changed, _] = sess.run([did_assignments_change, do_updates])

    [centers, assignments] = sess.run([centroids, cluster_assignments])
    end = time.time()
    print (("Found in %.2f seconds" % (end-start)), iters, "iterations")
    cluster_df=pd.DataFrame({'movies':df.index.values,'clusters':assignments})
    cluster_df.to_csv("clusters.csv",index=True)
    return assignments,df.index.values

# Read the main file i.e. ratings.dat
ratings_df = readers.read_file(FLAGS.data_file, sep="::")
clusters,movies = k_mean_clustering(ratings_df,K=FLAGS.K,MAX_ITERS = FLAGS.MAX_ITERS,TRAINED=FLAGS.TRAINED)

user_item=pd.read_pickle("user_item_table.pkl")
cluster=pd.read_csv("clusters.csv", index_col=False)

user_item=user_item.T

pcs = PCA(n_components=2, svd_solver='full')
cluster['x']=pcs.fit_transform(user_item)[:,0]
cluster['y']=pcs.fit_transform(user_item)[:,1]

fig = plt.figure()
ax = plt.subplot(111)


ax.scatter(cluster[cluster['clusters']==0]['x'].values,cluster[cluster['clusters']==0]['y'].values,color="r", label='cluster 0')
ax.scatter(cluster[cluster['clusters']==1]['x'].values,cluster[cluster['clusters']==1]['y'].values,color="g", label='cluster 1')
ax.scatter(cluster[cluster['clusters']==2]['x'].values,cluster[cluster['clusters']==2]['y'].values,color="b", label='cluster 2')
ax.scatter(cluster[cluster['clusters']==3]['x'].values,cluster[cluster['clusters']==3]['y'].values,color="k", label='cluster 3')
ax.scatter(cluster[cluster['clusters']==4]['x'].values,cluster[cluster['clusters']==4]['y'].values,color="c", label='cluster 4')

ax.legend()
plt.title("K-mean clustring")
plt.ylabel('PC2')
plt.xlabel('PC1');
plt.show()
