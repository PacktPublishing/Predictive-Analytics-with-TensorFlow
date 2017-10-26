# Imports for data io operations
from collections import deque
from six import next
import readers
import os

# Main imports for training
import tensorflow as tf
import numpy as np
import model as md
import pandas as pd

# Evaluate train times per epoch
import time

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

# Set random seed: for reproducibility
np.random.seed(12345)

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_file", "Input/ratings.dat", "Input user-movie-rating information file")
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1000, "Batch Size (default: 100)")
tf.flags.DEFINE_integer("dims", 15, "Dimensions of SVD (default: 15)")
tf.flags.DEFINE_integer("max_epochs", 50, "Dimensions of SVD (default: 25)")
tf.flags.DEFINE_string("checkpoint_dir", "save/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("val", True, "True if Folders with files and False if single file")
tf.flags.DEFINE_boolean("is_gpu", True, "Want to train model with GPU")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Remove all previous information saved or stored files
print("Start removing previous Files ...")
if os.path.isfile("model/user_item_table.pkl"):
    os.remove("model/user_item_table.pkl")
if os.path.isfile("model/user_item_table_train.pkl"):
    os.remove("model/user_item_table_train.pkl")
if os.path.isfile("model/item_item_corr.pkl"):
    os.remove("model/item_item_corr.pkl")
if os.path.isfile("model/item_item_corr_train.pkl"):
    os.remove("model/item_item_corr_train.pkl")
if os.path.isfile("model/user_user_corr.pkl"):
    os.remove("model/user_user_corr.pkl")
if os.path.isfile("model/user_user_corr_train.pkl"):
    os.remove("model/user_user_corr_train.pkl")
if os.path.isfile("model/clusters.csv"):
    os.remove("model/clusters.csv")
if os.path.isfile("model/val_error.pkl"):
    os.remove("model/val_error.pkl")
print("Done ...")


# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, "model")
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)


batch_size = FLAGS.batch_size # Number of samples per batch
dims = FLAGS.dims         # Dimensions of the data, 15
max_epochs = FLAGS.max_epochs   # Number of times the network sees all the training data

# Device used for all computations
if FLAGS.is_gpu:
    place_device = "/gpu:0"
else:
    place_device="/cpu:0"


def get_data():
    # Reads file using the demiliter :: form the ratings file
    # Download movie lens data from: http://files.grouplens.org/datasets/movielens/ml-1m.zip
    # Columns are user ID, item ID, rating, and timestamp
    # Sample data - 3::1196::4::978297539
    print("Inside get data ...")
    df = readers.read_file(FLAGS.data_file, sep="::")
    rows = len(df)

    # Purely integer-location based indexing for selection by position
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)

    # Split data into train and test, 75% for train and 25% for test
    split_index = int(rows * 0.75)

    # Use indices to separate the data
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)

    print("Done !!!")
    print(df.shape)
    return df_train, df_test,df['user'].max(),df['item'].max()

# Clip (limit) the values in an array: given an interval, values outside the interval are clipped to the interval edges. 
#For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
def clip(x):
    return np.clip(x, 1.0, 5.0) # rating 1 to 5

# Read data from ratings file to build a TF model
df_train, df_test,u_num,i_num = get_data()

print(u_num)
print(i_num)
u_num = 6040 # Number of users in the dataset
i_num = 3952 # Number of movies in the dataset


samples_per_batch = len(df_train) // batch_size
print("Number of train samples %d, test samples %d, samples per batch %d" % (len(df_train), len(df_test), samples_per_batch))

# Using a shuffle iterator to generate random batches, for training this helps preventing biased result
iter_train = readers.ShuffleIterator([df_train["user"],
                                     df_train["item"],
                                     df_train["rate"]],
                                     batch_size=batch_size)

# Sequentially generate one-epoch batches, for testing
iter_test = readers.OneEpochIterator([df_test["user"],
                                     df_test["item"],
                                     df_test["rate"]],
                                     batch_size=-1)

user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
rate_batch = tf.placeholder(tf.float32, shape=[None])

infer, regularizer = md.model(user_batch, item_batch, user_num=u_num, item_num=i_num, dim=dims, device=place_device)
_, train_op = md.loss(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=place_device)


saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
session_conf = tf.ConfigProto(
  allow_soft_placement=FLAGS.allow_soft_placement,
  log_device_placement=FLAGS.log_device_placement)

with tf.Session(config = session_conf) as sess:
    sess.run(init_op)
    print("%s\t%s\t%s\t%s" % ("Epoch", "Train err", "Validation err", "Elapsed Time"))
    errors = deque(maxlen=samples_per_batch)
    train_error=[]
    val_error=[]
    start = time.time()
    for i in range(max_epochs * samples_per_batch):
        users, items, rates = next(iter_train)
        _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users, item_batch: items, rate_batch: rates})
        pred_batch = clip(pred_batch)
        errors.append(np.power(pred_batch - rates, 2))
        if i % samples_per_batch == 0:
            train_err = np.sqrt(np.mean(errors))
            test_err2 = np.array([])
            for users, items, rates in iter_test:
                pred_batch = sess.run(infer, feed_dict={user_batch: users, item_batch: items})
                pred_batch = clip(pred_batch)
                test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
            end = time.time()

            print("%02d\t%.3f\t\t%.3f\t\t%.3f secs" % (i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)), end - start))
            train_error.append(train_err)
            val_error.append(np.sqrt(np.mean(test_err2)))
            start = end

    saver.save(sess, checkpoint_prefix)
    pd.DataFrame({'training error':train_error,'validation error':val_error}).to_pickle("val_error.pkl")
    print("Training Done !!!")

sess.close()

error = pd.read_pickle("val_error.pkl")
error.plot(title="Training vs validation error (per epoch)")
plt.ylabel('Error/loss')
plt.xlabel('Epoch');
plt.show()

if FLAGS.val:
    # Inference using saved model
    print("Validation ...")
    init_op = tf.global_variables_initializer()
    #checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    with tf.Session(config = session_conf) as sess:
        #sess.run(init_op)
        new_saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_prefix))
        new_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        test_err2 = np.array([])
        for users, items, rates in iter_test:
            pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                    item_batch: items})
            pred_batch = clip(pred_batch)
            test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
            print("Validation Error: ",np.sqrt(np.mean(test_err2)))
    print("Done !!!")
sess.close()

def create_df(ratings_df=readers.read_file(FLAGS.data_file, sep="::")):
    """
    Use to create a trained DataFrame,all missing values in user-item table
    is filled here using SVD trained model
    INPUTS :
        ratings_df : rating dataframe, store all users rating for respective movies

    OUTPUT:
        Filled rating dataframe where user is row and item is col
    """
    if os.path.isfile("model/user_item_table.pkl"):
        df=pd.read_pickle("user_item_table.pkl")
    else:
        df = ratings_df.pivot(index = 'user', columns ='item', values = 'rate').fillna(0)
        df.to_pickle("user_item_table.pkl")
    
    df=df.T
    users=[]
    items=[]
    start = time.time()

    print("Start creating user-item dense table")
    total_movies=list(ratings_df.item.unique())
    for index in df.columns.tolist():
        #rated_movies=ratings_df[ratings_df['user']==index].drop(['st', 'user'], axis=1)
        rated_movie=[]
        rated_movie=list(ratings_df[ratings_df['user']==index].drop(['st', 'user'], axis=1)['item'].values)
        unseen_movies=[]
        unseen_movies=list(set(total_movies) - set(rated_movie))
        for movie in unseen_movies:
            users.append(index)
            items.append(movie)

    end = time.time()
    print(("Found in %.2f seconds" % (end-start)))
    del df
    rated_list = []

    init_op = tf.global_variables_initializer()
    #checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    with tf.Session(config = session_conf) as sess:
        #sess.run(init_op)
        print("prediction started ...")
        new_saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_prefix))
        new_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        test_err2 = np.array([])
        rated_list = sess.run(infer, feed_dict={user_batch: users, item_batch: items})
        rated_list = clip(rated_list)
        print("Done !!!")

    sess.close()
    df_dict={'user':users,'item':items,'rate':rated_list}
    df = ratings_df.drop(['st'],axis=1).append(pd.DataFrame(df_dict)).pivot(index = 'user', columns ='item', values = 'rate').fillna(0)
    df.to_pickle("user_item_table_train.pkl")
    return df

create_df(ratings_df = readers.read_file(FLAGS.data_file, sep="::"))
