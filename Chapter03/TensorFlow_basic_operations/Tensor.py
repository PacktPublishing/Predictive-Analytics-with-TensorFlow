import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()


X = [[2.0, 4.0],
    [6.0, 8.0]]
Y = np.array([[2.0, 4.0],
            [6.0, 6.0]], dtype=np.float32)
Z = tf.constant([[2.0, 4.0],
                [6.0, 8.0]])

print(type(X))
print(type(Y))
print(type(Z))

t1 = tf.convert_to_tensor(X, dtype=tf.float32)
t2 = tf.convert_to_tensor(Z, dtype=tf.float32)
t3 = tf.convert_to_tensor(Z, dtype=tf.float32)

print(type(t1))
print(type(t2))
print(type(t3))
