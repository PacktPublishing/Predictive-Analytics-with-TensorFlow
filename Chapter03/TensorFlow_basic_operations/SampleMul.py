import tensorflow as tf
import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

x = tf.constant(8)
y = tf.constant(9)
z = tf.multiply(x, y)

sess = tf.Session()
out_z = sess.run(z)

print('The multiplicaiton of x and y: %d' % out_z)
