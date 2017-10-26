import tensorflow as tf
import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b)
sess = tf.Session()
print(sess.run(y, feed_dict={a: 3, b: 5}))
