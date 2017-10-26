import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.cross_validation import train_test_split
import warnings
from tensorflow.python.framework import ops
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

FILE_PATH = '../input/bank_normalized.csv'	                            # Path to .csv dataset
raw_data = pd.read_csv(FILE_PATH)					    # Open raw .csv

print("Raw data loaded successfully...\n")
Y_LABEL = 'y'                                   			    # Name of the variable to be predicted
KEYS = [i for i in raw_data.keys().tolist() if i != Y_LABEL]	            # Name of predictors
N_INSTANCES = raw_data.shape[0]                     			    # Number of instances
N_INPUT = raw_data.shape[1] - 1                     			    # Input size
N_CLASSES = raw_data[Y_LABEL].unique().shape[0]     			    # Number of classes (output size)
TEST_SIZE = 0.25                                    			    # Test set size (% of dataset)
TRAIN_SIZE = int(N_INSTANCES * (1 - TEST_SIZE))     			    # Train size
LEARNING_RATE = 0.001                               			    # Learning rate
TRAINING_EPOCHS = 10                             			    # Number of epochs
BATCH_SIZE = 100                                   			    # Batch size
DISPLAY_STEP = 20                                    			    # Display progress each x epochs
HIDDEN_SIZE = 256	                                   		    # Number of hidden neurons 256
ACTIVATION_FUNCTION_OUT = tf.nn.tanh                                        # Last layer act fct
STDDEV = 0.1                                        			    # Standard deviation (for weights random init)
RANDOM_STATE = 100						            # Random state for train_test_split

print("Variables loaded successfully...\n")
print("Number of predictors \t%s" %(N_INPUT))
print("Number of classes \t%s" %(N_CLASSES))
print("Number of instances \t%s" %(N_INSTANCES))
print("\n")   

# Load data
data = raw_data[KEYS].get_values()                  			# X data
labels = raw_data[Y_LABEL].get_values()             			# y data

# One hot encoding for labels
labels_ = np.zeros((N_INSTANCES, N_CLASSES))
labels_[np.arange(N_INSTANCES), labels] = 1

# Train-test split
data_train, data_test, labels_train, labels_test = train_test_split(data,labels_,test_size = TEST_SIZE,random_state = RANDOM_STATE)

print("Data loaded and splitted successfully...\n")
#------------------------------------------------------------------------------
# Neural net construction

# Net params
n_input = N_INPUT                   # input n labels
n_hidden_1 = HIDDEN_SIZE            # 1st layer
n_hidden_2 = HIDDEN_SIZE            # 2nd layer
n_hidden_3 = HIDDEN_SIZE            # 3rd layer
n_hidden_4 = HIDDEN_SIZE            # 4th layer
n_classes = N_CLASSES               # output m classes

# TensorFlow placeholders
X = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
dropout_keep_prob = tf.placeholder(tf.float32)


def DeepMLPClassifier(_X, _weights, _biases, dropout_keep_prob):
    layer1 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), dropout_keep_prob)
    layer2 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2'])), dropout_keep_prob)
    layer3 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer2, _weights['h3']), _biases['b3'])), dropout_keep_prob)
    layer4 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer3, _weights['h4']), _biases['b4'])), dropout_keep_prob)
    out = ACTIVATION_FUNCTION_OUT(tf.add(tf.matmul(layer4, _weights['out']), _biases['out']))
    return out

# Here are the dictionary of weights and biases of each layer
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=STDDEV)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=STDDEV)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],stddev=STDDEV)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],stddev=STDDEV)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes],stddev=STDDEV)),                                   
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Build model
pred = DeepMLPClassifier(X, weights, biases, dropout_keep_prob)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

# Accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                                    
print("Deep MLP networks has been built successfully...")
print("Starting training...")
#------------------------------------------------------------------------------

# Initialize variables
init_op = tf.global_variables_initializer()

# Launch session
sess = tf.Session()
sess.run(init_op)

acc_list = []
cost_list = []
i_data = []

# Training loop
for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0.0    
    total_batch = int(data_train.shape[0] / BATCH_SIZE)
    # Loop over all batches
    for i in range(total_batch):
        randidx = np.random.randint(int(TRAIN_SIZE), size = BATCH_SIZE)
        batch_xs = data_train[randidx, :]
        batch_ys = labels_train[randidx, :]
        # Fit using batched data
        sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 0.9})
        # Calculate average cost
        avg_cost += sess.run(cost, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob:1.})/total_batch  
    # Display progress
    if epoch % DISPLAY_STEP == 0:
        i_data.append(epoch+1)
        cost_list.append(avg_cost)
        print ("Epoch:%3d/%3d, cost:%.9f" % (epoch, TRAINING_EPOCHS, avg_cost))
        train_acc = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob:1.})
        acc_list.append(train_acc)
        print ("Training accuracy: %.3f" % (train_acc))


print("Your deep MLP model has been trained sucessfully.")
print("Evaluating deep MLP on the test set...")
#------------------------------------------------------------------------------
# Testing

test_acc = sess.run(accuracy, feed_dict={X: data_test, y: labels_test, dropout_keep_prob:1.})
print ("Prediction/clasification accuracy: %.3f" % (test_acc))


# Plot loss over time
plt.subplot(221)
plt.plot(i_data, cost_list, 'k--', label='Training loss', linewidth=1.0)
plt.title('Cross entropy loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('Cross entropy loss')
plt.legend(loc='upper right')
plt.grid(True)

with open("i_data.txt", "w") as output:
    output.write(str(i_data))

with open("cost_list.txt", "w") as output:
    output.write(str(cost_list))

# Plot train and test accuracy
plt.subplot(222)
plt.plot(i_data, acc_list, 'r--', label='Accuracy on the training set', linewidth=1.0)
plt.title('Accuracy on the training set')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

with open("acc_list.txt", "w") as output:
    output.write(str(acc_list))

sess.close()
print("Session closed!")

