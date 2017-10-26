import os
import numpy as np # For Python NumPy array
import csv # For reading and parsing CSV files
import string
import requests # Python HTTP for human
import io
from zipfile import ZipFile
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('punkt')

# Normalize text -e.g. making them lower case
texts = ['Maybe, i should get some coffee before starting at 21  in the ']
texts = [x.lower() for x in texts]
print(texts)

#Remove punctuation
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
print(texts)

# Remove numbers
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
print(texts)

# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]
print(texts)

#Sto wirds removal
texts = 'Maybe i should get some coffee before starting'
stop = set(['i', 'should', 'some', 'before'])
texts = [i for i in texts.lower().split() if i not in stop]
print(texts)

print(type(texts))
filtered_texts = ' '.join(texts)
print(filtered_texts)

# Plot histogram of text lengths
text_lengths = [len(x.split()) for x in texts]
text_lengths = [x for x in text_lengths if x < 50]

max_features = 10

# Define tokenizer
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

# Create TF-IDF of texts
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(texts)
print(sparse_tfidf_texts)

