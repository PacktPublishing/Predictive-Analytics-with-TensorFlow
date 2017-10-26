import tensorflow as tf

x = tf.constant([[2., 4., 6., 8.,],
                 [10., 12., 14., 16.]])

x = tf.reshape(x, [1, 2, 4, 1])  # give a shape accepted by tf.nn.max_pool

VALID = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
SAME = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

print(VALID.get_shape()) 
print(SAME.get_shape()) 

'''
>>>
(1, 1, 2, 1)
(1, 1, 2, 1)
'''
