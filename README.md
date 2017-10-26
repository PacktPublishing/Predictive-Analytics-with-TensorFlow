# Predictive Analytics with TensorFlow
This is the code repository for [Predictive Analytics with TensorFlow](https://www.packtpub.com/big-data-and-business-intelligence/predictive-analytics-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781788398923), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.
## About the Book
Predictive decisions are becoming a huge trend worldwide catering wide sectors of industries by predicting which decisions are more likely to give maximum results. The data mining, statistics, machine learning allows users to discover predictive intelligence by uncovering patterns and showing the relationship among the structured and unstructured data. This book will help you build solutions which will make automated decisions. In the end tune and build your own predictive analytics model with the help of TensorFlow.

This book will be divided in three main sections.

In the first section-Applied Mathematics, Statistics, and Foundations of Predictive Analytics; will cover Linear algebra needed to getting started with data science in a practical manner by using the most commonly used Python packages. It will also cover the needed background in probability and information theory that is must for Data Scientists.

The second section shows how to develop large-scale predictive analytics pipelines using supervised (classification/regression) and unsupervised (clustering) learning algorithms. It’ll then demonstrate how to develop predictive models for NLP. Finally, reinforcement learning and recommendation system will be used for developing predictive models.

The third section covers practical mastery of deep learning architectures for advanced predictive analytics: including Deep Neural Networks (MLP & DBN) and Recurrent Neural Networks for high-dimensional and sequence data. Finally, it’ll show how to develop Convolutional Neural Networks- based predictive models for emotion recognition, image classification, and sentiment analysis.

So in total, this book will help you control the power of deep learning in diverse fields, providing best practices and tips from the real world use cases and helps you in decision making based on predictive analytics.

## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.



The code will look like the following:
```
import tensorflow as tf
import numpy as np

raw_data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)
update_avg = alpha * curr_value + (1 - alpha) * prev_avg
```



## Related Products
* [Machine Learning with TensorFlow](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781786462961)

* [Hands-On Deep Learning with TensorFlow](https://www.packtpub.com/big-data-and-business-intelligence/hands-deep-learning-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781787282773)

* [Deep Learning with TensorFlow](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781786469786)

### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSe5qwunkGf6PUvzPirPDtuy1Du5Rlzew23UBp2S-P3wB-GcwQ/viewform) if you have any feedback or suggestions.
