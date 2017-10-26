import tensorflow as tf

X = [1.5, 0.5, 0.75, 1.0, 0.75, 0.6, 0.4, 0.9]
drop_out = tf.nn.dropout(X, 0.5)

sess = tf.Session()
with sess.as_default():
	print(drop_out.eval())
	
sess.close()	
