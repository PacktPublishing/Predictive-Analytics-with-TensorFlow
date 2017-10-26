# Using Word2Vec for prediction: In this example, we will load the CBOW trained embeddings to perform movie review predictions using LR model. From this dataset we will compute/fit the CBOW model using the Word2Vec algorithm
# -----------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
#import urllib.request
import preprocessor
from nltk.corpus import stopwords
from tensorflow.python.framework import ops
import warnings
import random
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Start a graph session
sess = tf.Session()

# Declare model parameters
batch_size = 500
embedding_size = 200
vocabulary_size = 15000
generations = 100000
model_learning_rate = 0.001
max_words = 100

# Declare stop words
stops = stopwords.words('english')

# Load Data
print('Loading Data... ')
data_folder_name = 'temp'
texts, target = preprocessor.load_movie_data()

# Normalize text
print('Normalizing Text Data... ')
texts = preprocessor.normalize_text(texts, stops)

# Texts must contain at least 4 words
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 3]
texts = [x for x in texts if len(x.split()) > 3]

# Split up data set into train/test
train_indices = np.random.choice(len(target), round(0.75*len(target)), replace=False)
test_indices = np.array(list(set(range(len(target))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# Load dictionary and embedding matrix
dict_file = 'temp/movie_vocab.pkl'
word_dictionary = pickle.load(open(dict_file, 'rb'))

# Convert texts to lists of indices
text_data_train = np.array(preprocessor.text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(preprocessor.text_to_numbers(texts_test, word_dictionary))

# Pad/crop movie reviews to specific length
text_data_train = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_train]])
text_data_test = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_test]])

print('Creating Model... ')
# Define Embeddings:
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# Define model:
# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Initialize placeholders
x_data = tf.placeholder(shape=[None, max_words], dtype=tf.int32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Lookup embeddings vectors
embed = tf.nn.embedding_lookup(embeddings, x_data)
# Take average of all word embeddings in documents
embed_avg = tf.reduce_mean(embed, 1)

# Declare logistic model (sigmoid in loss function)
model_output = tf.add(tf.matmul(embed_avg, A), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Declare optimizer
training_op = tf.train.GradientDescentOptimizer(0.001)
train_step = training_op.minimize(loss)

# Intitialize Variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Load model embeddings
model_checkpoint_path = 'temp/cbow_movie_embeddings.ckpt'
saver = tf.train.Saver({"embeddings": embeddings})
saver.restore(sess, model_checkpoint_path)


# Start Logistic Regression
print('Starting Model Training... ')
train_loss = [] 
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(3000):
    rand_index = np.random.choice(text_data_train.shape[0], size=batch_size)
    rand_x = text_data_train[rand_index]
    rand_y = np.transpose([target_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    # Only record loss and accuracy every 100 generations
    if (i+1)%100==0:
        i_data.append(i+1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)
        
        test_loss_temp = sess.run(loss, feed_dict={x_data: text_data_test, y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)
        
        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)
    
        test_acc_temp = sess.run(accuracy, feed_dict={x_data: text_data_test, y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)
    if (i+1)%500==0:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Iteration # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

print('\nOverall accuracy on test set (%): {}'.format(np.mean(test_acc)*100.0))

# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Training loss')
plt.plot(i_data, test_loss, 'r--', label='Test loss', linewidth=4)
plt.title('Cross entropy loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('Cross entropy loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Accuracy on the training set')
plt.plot(i_data, test_acc, 'r--', label='Accuracy on the test set', linewidth=4)
plt.title('Accuracy on the train and test set')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
