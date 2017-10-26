# Word2Vec: CBOW Model (Continuous Bag of Words): In this example, we will download and preprocess the movie review data. 
# From this data set we will compute/fit the CBOW model of the Word2Vec Algorithm

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
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

# Make a saving directory if it doesn't exist
data_folder_name = 'temp'
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

# Start a graph session
sess = tf.Session()

# Declare model parameters
batch_size = 500
embedding_size = 200
vocabulary_size = 15000
iterations= 5000
learning_rate = 0.001

num_sampled = int(batch_size/2)    # Number of negative examples to sample.
window_size = 4       # How many words to consider left and right.

# Add checkpoints to training
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100

# Declare stop words
stops = stopwords.words('english')

# We pick some test words. We are expecting synonyms to appear
# Below words adopted from http://member.tokoha-u.ac.jp/~dixonfdm/Writing%20Topics%20htm/Movie%20Review%20Folder/movie_descrip_vocab.htm
valid_words = ['love', 'romance', 'sad', 'average', 'boring', 'entertaining', 'exciting', 'excellent', 'fantastic', 'action', 'thriller', 'horror', 'adventure', 'suspenseful']
# Later we will have to transform these into indices

# Load the movie review data
print('Loading Data ... ')
texts, target = preprocessor.load_movie_data()

# Normalize text
print('Normalizing Text Data ... ')
texts = preprocessor.normalize_text(texts, stops)

# Texts must contain at least 4 words
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 3]
texts = [x for x in texts if len(x.split()) > 3]    

# Build our data set and dictionaries
print('Creating Dictionary ... ')
word_dictionary = preprocessor.build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = preprocessor.text_to_numbers(texts, word_dictionary)

# Get validation word keys
valid_examples = [word_dictionary[x] for x in valid_words]    

print('Creating Model ... ')
# Define Embeddings:
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Create data/target placeholders
x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*window_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Lookup the word embedding
# Add together window embeddings:
embed = tf.zeros([batch_size, embedding_size])
for element in range(2*window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

# Get loss from prediction
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))
                                     
# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Cosine similarity between words
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Create model saving operation
saver = tf.train.Saver({"embeddings": embeddings})

#Add variable initializer.
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Filter out sentences that aren't long enough:
text_data = [x for x in text_data if len(x)>=(2*window_size+1)]

# Run the CBOW model.
print('Model Training ... ')
loss_vec = []
loss_x_vec = []
for i in range(iterations):
    batch_inputs, batch_labels = preprocessor.generate_batch_data(text_data, batch_size, window_size, method='cbow')
    feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}

    # Run the train step
    sess.run(optimizer, feed_dict=feed_dict)
            
    # Save dictionary + embeddings
    if (i+1) % save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join(data_folder_name,'movie_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)
        
        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(),data_folder_name,'cbow_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))
