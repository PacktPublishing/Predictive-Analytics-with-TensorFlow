from scipy import misc
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
import os, sys, inspect
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import Utility
from Utility import testResult

from tensorflow.python.framework import ops
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

emotion = {0:'anger', 1:'disgust',\
           2:'fear',3:'happy',\
           4:'sad',5:'surprise',6:'neutral'}

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('test_photos/happy.jpg')     
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "input/", "Path to data files (train and test)")
tf.flags.DEFINE_string("logs_dir", "Logs/CNN_logs/", "Logging path")
tf.flags.DEFINE_string("mode", "train", "mode: train (Default)/ test")

train_images, train_labels, valid_images, valid_labels, test_images = Utility.read_data(FLAGS.data_dir)

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph('Logs/CNN_logs/model.ckpt-900.meta')
new_saver.restore(sess, 'Logs/CNN_logs/model.ckpt-900')
tf.get_default_graph().as_graph_def()

x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")

image_0 = np.resize(gray,(1,48,48,1))
tResult = testResult()
num_evaluations = 5000

for i in range(0,num_evaluations):
	result = sess.run(y_conv, feed_dict={x:image_0})
	label = sess.run(tf.argmax(result, 1))
	label = label[0]
	label = int(label)
	tResult.evaluate(label)
tResult.display_result(num_evaluations)
