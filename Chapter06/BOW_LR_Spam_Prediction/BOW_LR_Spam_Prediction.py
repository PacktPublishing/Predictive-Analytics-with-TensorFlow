# Working with Bag of Words
#---------------------------------------
# In this example, we will download and preprocess the ham/spam text data. We will then use a one-hot-encoding to make a
# bag of words set of features to use in logistic regression. We will use these one-hot-vectors for logistic regression to
# predict if a text is spam or ham.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import string
import requests
import io
from zipfile import ZipFile
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split

from tensorflow.python.framework import ops
import warnings
import random
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Check if data was downloaded, otherwise download it and save for future use
save_file_name = os.path.join('temp','temp_spam_data.csv')

# Create directory if it doesn't exist
if not os.path.exists('temp'):
    os.makedirs('temp')

if os.path.isfile(save_file_name):
    text_data = []
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            text_data.append(row)
else:
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii',errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x)>=1]
    
    # And write to csv
    with open(save_file_name, 'w') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data)

texts = [x[1] for x in text_data]
label = [x[0] for x in text_data]

# Relabel 'spam' as 1, 'ham' as 0
target = [1 if x=='spam' else 0 for x in label]

# Normalize text -e.g. making them lower case
texts = [x.lower() for x in texts]

# Remove punctuation
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

# Remove numbers
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]

# Plot histogram of text lengths
text_lengths = [len(x.split()) for x in texts]
text_lengths = [x for x in text_lengths if x < 50]
plt.hist(text_lengths, bins=50)
plt.title('Histogram of # of words in texts')
plt.xlabel('Number of words')
plt.ylabel('Word frequency')
plt.show()

# Choose max text word length at 25
text_size = 25
min_word_freq = 10

# Setup vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor(text_size, min_frequency=min_word_freq)

# Have to fit transform to get length of unique words.
embed_vocab = vocab_processor.fit_transform(texts)
embedding_size = len(vocab_processor.vocabulary_)

# Split up data set into train/test

texts_train, texts_test, target_train, target_test = train_test_split(texts, target, train_size = 0.75) 

'''
train_indices = np.random.choice(len(texts), round(len(texts)*0.75), replace=False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = [x for ix, x in enumerate(target) if ix in train_indices]
target_test = [x for ix, x in enumerate(target) if ix in test_indices]
'''

# Setup Index Matrix for one-hot-encoding
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Initialize placeholders
x_data = tf.placeholder(shape=[text_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)

# Text-Vocab Embedding
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_sums = tf.reduce_sum(x_embed, 0)

# Declare model operations
x_sums_2D = tf.expand_dims(x_sums, 0)
model_output = tf.add(tf.matmul(x_sums_2D, A), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Prediction operation
prediction = tf.sigmoid(model_output)

# Declare optimizer
train_op = tf.train.GradientDescentOptimizer(0.001)
#  train_op = tf.train.AdamOptimizer(0.01)
train_step = train_op.minimize(loss)

# Intitialize Variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Start Logistic Regression
print('Starting training over {} sentences: '.format(len(texts_train)))
loss_vec_train = []
train_acc_all = []
train_acc_avg = []
for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
    y_data = [[target_train[ix]]]    
    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data}) 
    loss_vec_train.append(temp_loss)    
    if (ix+1)%10==0:
        print('Training Observation: ' + str(ix+1) + ', Loss = ' + str(temp_loss))        
    # Keep trailing average of past 75 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})
    # Get True/False if prediction is accurate
    train_acc_temp = target_train[ix]==np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))

# Plotting the error in training
with warnings.catch_warnings():
	warnings.simplefilter("ignore", category=RuntimeWarning)
	plt.plot([np.mean(loss_vec_train[i-50:i]) for i in range(len(loss_vec_train))]) 
	plt.xlabel('Training observation')
	plt.ylabel('Training loss')
	plt.show()

# Evaluating the model on test set
print('Test set accuracy over {} sentences:- '.format(len(texts_test)))
test_acc_all = []
for ix, t in enumerate(vocab_processor.transform(texts_test)):
    y_data = [[target_test[ix]]]    
    #if (ix+1)%10==0:
        #print('Test observation: ' + str(ix+1))    
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})   
    # Get True/False if prediction is accurate
    test_acc_temp = target_test[ix]==np.round(temp_pred)
    test_acc_all.append(test_acc_temp)   

print('\nOverall accuracy on test set (%): {}'.format(np.mean(test_acc_all)*100.0))

number_of_correct_predicted_words = len(texts_test) - np.sum(test_acc_all)
print('Number of wrongly predicted texts on test set: {}'.format(number_of_correct_predicted_words))

# Plot training accuracy over time
plt.plot(range(len(train_acc_avg)), train_acc_avg, 'r-', label='Training accuracy')
plt.title('Average training accuracy over 75 iterations')
plt.xlabel('Iteration')
plt.ylabel('Training accuracy')
plt.show()
