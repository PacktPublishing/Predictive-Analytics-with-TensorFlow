def conv_layer(data, weights, bias, strides=1):
	x = tf.nn.conv2d(x, 
					weights, 
					strides=[1, strides, strides, 1], 
					padding='SAME')
	x = tf.nn.bias_add(data, bias)
	return tf.nn.relu(x)
