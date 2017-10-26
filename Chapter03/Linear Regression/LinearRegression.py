# Import libraries (Numpy, Tensorflow, matplotlib)
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf

import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

# Create 1000 points following a function y=0.1 * x + 0.4 (i.e. y = W * x + b) with some normal random distribution
num_points = 1000
vectors_set = []
for i in range(num_points):
    W = 0.1  # W
    b = 0.4  # b
    x1 = np.random.normal(0.0, 1.0)
    nd = np.random.normal(0.0, 0.05)
    y1 = W * x1 + b
    # Add some impurity with the some normal distribution -i.e. nd
    y1 = y1 + nd
    # Append them and create a combined vector set
    vectors_set.append([x1, y1])

# Seperate the data point across axixes
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# Plot and show the data points on a 2D space
plot.plot(x_data, y_data, 'ro', label='Original data')
plot.legend()
plot.show()

#tf.name_scope organize things on the tensorboard graph view
with tf.name_scope("LinearRegression") as scope:
	W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="Weights")
	b = tf.Variable(tf.zeros([1]))
	y = W * x_data + b

# Define a loss function that take into account the distance between the prediction and our dataset
with tf.name_scope("LossFunction") as scope:
	loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.6)
train = optimizer.minimize(loss)


# Annotate loss, weights and bias (Needed for tensorboard)
loss_summary = tf.summary.scalar("loss", loss)
w_ = tf.summary.histogram("W", W)
b_ = tf.summary.histogram("b", b)

# Merge all the summaries
merged_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Writer for tensorboard (Directory)
writer_tensorboard = tf.summary.FileWriter('logs/', tf.get_default_graph())

for i in range(16):
	sess.run(train)
	print(i, sess.run(W), sess.run(b), sess.run(loss))
	plot.plot(x_data, y_data, 'ro', label='Original data')
	plot.plot(x_data, sess.run(W)*x_data + sess.run(b))
	plot.xlabel('X')
	plot.xlim(-2, 2)
	plot.ylim(0.1, 0.6)
	plot.ylabel('Y')
	plot.legend()
	plot.show()
