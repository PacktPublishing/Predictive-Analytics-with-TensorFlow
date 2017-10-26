import matplotlib.pyplot as plt
import numpy as np
import random
import os
import tensorflow as tf
import requests
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

# Load the data
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
num_features = len(housing_header)
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]

y_vals = np.transpose([np.array([y[len(housing_header)-1] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in housing_header] for y in housing_data])

## Min-Max Scaling
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

# Split the data into train and test sets
random.seed(12345)
train_indices = np.random.choice(len(x_vals), int(len(x_vals)*0.75), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare the batch size
batch_size=len(x_vals_test)

# Placeholders
x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare distance metric: L1
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2)

def kNN(k): 
    # Predict: Get min distance index (Nearest neighbor)
    topK_X, topK_indices = tf.nn.top_k(tf.negative(distance), k=k)
    x_sums = tf.expand_dims(tf.reduce_sum(topK_X, 1), 1)
    x_sums_repeated = tf.matmul(x_sums,tf.ones([1, k], tf.float32))
    x_val_weights = tf.expand_dims(tf.div(topK_X, x_sums_repeated), 1)

    topK_Y = tf.gather(y_target_train, topK_indices)
    prediction = tf.squeeze(tf.matmul(x_val_weights,topK_Y), axis=[1])

    # Calculate MSE
    mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

    # Calculate how many loops over training data
    num_loops = int(np.ceil(len(x_vals_test)/batch_size))

    # Initialize the global variables
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
            sess.run(init_op) 
            for i in range(num_loops):
                min_index = i*batch_size
                max_index = min((i+1)*batch_size,len(x_vals_train))
                x_batch = x_vals_test[min_index:max_index]
                y_batch = y_vals_test[min_index:max_index]
                predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})
                batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})           
    return batch_mse

mse_list = []
k_list = []
def getOptimalMSE_K():
    mse = 0.0
    for k in range(2, 11):
        mse = kNN(k)  
        mse_list.append(mse)
        k_list.append(k)
    return k_list, mse_list 

k_list, mse_list  = getOptimalMSE_K()
dict_list = zip(k_list, mse_list)
my_dict = dict(dict_list)
print(my_dict)
optimal_k = min(my_dict, key=my_dict.get)

print("Optimal K value: ", optimal_k)
mse = min(mse_list)
#Calculate and print: mse, accuracy
#mse = np.round(batch_mse)
print("Minimum mean square error: ", mse)

def bestKNN(k): 
    # Predict: Get min distance index (Nearest neighbor)
    topK_X, topK_indices = tf.nn.top_k(tf.negative(distance), k=k)
    x_sums = tf.expand_dims(tf.reduce_sum(topK_X, 1), 1)
    x_sums_repeated = tf.matmul(x_sums,tf.ones([1, k], tf.float32))
    x_val_weights = tf.expand_dims(tf.div(topK_X, x_sums_repeated), 1)

    topK_Y = tf.gather(y_target_train, topK_indices)
    prediction = tf.squeeze(tf.matmul(x_val_weights,topK_Y), axis=[1])

    # Calculate how many loops over training data
    num_loops = int(np.ceil(len(x_vals_test)/batch_size))

    # Initialize the global variables
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
            sess.run(init_op) 
            for i in range(num_loops):
                min_index = i*batch_size
                max_index = min((i+1)*batch_size,len(x_vals_train))
                x_batch = x_vals_test[min_index:max_index]
                y_batch = y_vals_test[min_index:max_index]
                predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})
    return predictions, y_batch

predicted_labels, actual_labels  = bestKNN(optimal_k)

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
	    if(np.round(testSet[x]) == np.round(predictions[x])):
                correct += 1
	return (correct/float(len(testSet))) * 100.0
accuracy = getAccuracy(actual_labels, predicted_labels)

print('Accuracy: ' + repr(accuracy) + '%')

# Plot prediction and actual distribution
bins = np.linspace(5, 50, 45)
plt.hist(predicted_labels, bins, alpha=1.0, facecolor='red', label='Prediction')
plt.hist(actual_labels, bins, alpha=1.0, facecolor='green', label='Actual')
plt.title('predicted vs actual values')
plt.xlabel('Median house price in $1,000s')
plt.ylabel('count')
plt.legend(loc='upper right')
plt.show()

