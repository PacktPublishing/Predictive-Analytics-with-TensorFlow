import math
from textblob import TextBlob as tb
import numpy as np

def tf(word, doc):
    return doc.words.count(word) / len(doc.words)

def n_contain(word, doclist):
    return sum(1 for doc in doclist if word in blob.words)

def idf(word, doclist):
    return math.log(len(doclist) / (1 + n_contain(word, doclist)))

def tfidf(word, doc, doclist):
    return tf(word, doc) * idf(word, doclist)

doc1 = tb("""TensorFlow is an open source software library for machine learning across a range of tasks, and developed by Google to meet their needs for systems capable of building and training neural networks to detect and decipher patterns and correlations, analogous to the learning and reasoning which humans use""")
doc2 = tb("""It is currently used for both research and production at Google products,‍ often replacing the role of its closed-source predecessor, DistBelief. TensorFlow was originally developed by the Google Brain team for internal Google use before being released under the Apache 2.0 open source license on November 9, 2015""")
doc3 = tb("""Starting in 2011, Google Brain built DistBelief as a proprietary machine learning system based on deep learning neural networks. Its use grew rapidly across diverse Alphabet companies in both research and commercial applications""")
blobList = [doc1, doc2, doc3]

for i, blob in enumerate(blobList):
    print("Top words in document {}".format(i+1))
    scores = {word: tfidf(word, blob, blobList) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:5]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 10)))
