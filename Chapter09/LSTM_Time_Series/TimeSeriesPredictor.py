import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import time_series_preprocessor as tsp
import matplotlib.pyplot as plt
from tensorflow.python.ops import metrics_impl

from tensorflow.python.framework import ops
import warnings
import random
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

# Hyperparameters
input_dim = 1
seq_size = 5
hidden_dim = 5

# Weight variables and input placeholders
W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
b_out = tf.Variable(tf.random_normal([1]), name='b_out')
x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
y = tf.placeholder(tf.float32, [None, seq_size])

def LSTM_Model():
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn_cell.BasicLSTMCell(hidden_dim)
        outputs, states = rnn.dynamic_rnn(cell, x, dtype=tf.float32)
        num_examples = tf.shape(x)[0]
        W_repeated = tf.tile(tf.expand_dims(W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + b_out
        out = tf.squeeze(out)
        return out

train_loss = []
test_loss = []
step_list = []
    
def trainNetwork(train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            max_patience = 3
            patience = max_patience
            min_test_err = float('inf')
            step = 0
            while patience > 0:
                _, train_err = sess.run([train_op, cost], feed_dict={x: train_x, y: train_y})
                if step % 100 == 0:
                    test_err = sess.run(cost, feed_dict={x: test_x, y: test_y})
                    print('step: {}\t\ttrain err: {}\t\ttest err: {}'.format(step, train_err, test_err))
                    train_loss.append(train_err)
                    test_loss.append(test_err) 
                    step_list.append(step)                   

                    if test_err < min_test_err:
                        min_test_err = test_err
                        patience = max_patience
                    else:
                        patience -= 1
                step += 1                
            save_path = saver.save(sess, 'model/model.ckpt')
            print('Model saved to {}'.format(save_path))

# Cost optimizer
cost = tf.reduce_mean(tf.square(LSTM_Model()- y))
train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(cost)

# Auxiliary ops
saver = tf.train.Saver()
def testLSTM(sess, test_x):
        tf.get_variable_scope().reuse_variables()
        saver.restore(sess, 'model/model.ckpt')
        output = sess.run(LSTM_Model(), feed_dict={x: test_x})
        return output

def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def main():    
    data = tsp.load_series('input/international-airline-passengers.csv')
    train_data, actual_vals = tsp.split_data(data=data, percent_train=0.75)

    train_x, train_y = [], []
    for i in range(len(train_data) - seq_size - 1):
        train_x.append(np.expand_dims(train_data[i:i+seq_size], axis=1).tolist())
        train_y.append(train_data[i+1:i+seq_size+1])

    test_x, test_y = [], []
    for i in range(len(actual_vals) - seq_size - 1):
        test_x.append(np.expand_dims(actual_vals[i:i+seq_size], axis=1).tolist())
        test_y.append(actual_vals[i+1:i+seq_size+1])

    trainNetwork(train_x, train_y, test_x, test_y)

    with tf.Session() as sess:
        predicted_vals = testLSTM(sess, test_x)[:,0]
        #print('predicted_vals', np.shape(predicted_vals))
        # Following prediction results of the model given ground truth values
        plot_results(train_data, predicted_vals, actual_vals, 'ground_truth_predition.png')

        prev_seq = train_x[-1]
        predicted_vals = []
        for i in range(100):
            next_seq = testLSTM(sess, [prev_seq])
            predicted_vals.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        # Following predictions results where only the training data was given
        plot_results(train_data, predicted_vals, actual_vals, 'prediction_on_train_set.png')

main()

def plot_error():
	# Plot loss over time
	plt.plot(step_list, train_loss, 'r--', label='LSTM training loss per iteration', linewidth=4)
	plt.title('LSTM training loss per iteration')
	plt.xlabel('Iteration')
	plt.ylabel('Training loss')
	plt.legend(loc='upper right')
	plt.show()

	# Plot accuracy over time
	plt.plot(step_list, test_loss, 'r--', label='LSTM test loss per iteration', linewidth=4)
	plt.title('LSTM test loss per iteration')
	plt.xlabel('Iteration')
	plt.ylabel('Test loss')
	plt.legend(loc='upper left')
	plt.show()

plot_error()
