import tensorflow as tf

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, 
			ksize=[1, k, k, 1], 
			strides=[1, k, k, 1], 
			padding='SAME')
