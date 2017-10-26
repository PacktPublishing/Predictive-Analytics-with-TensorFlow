# Basic libraries
import numpy as np
import pandas as pd
from scipy import stats
import math

# Machine Learning
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.python.framework import ops
import warnings
import random
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('input/fandango_score_comparison.csv')
print(df.head())

# WATCH OUT for the following typo: 'Metacritic_user_nom' and it should have been actually 'Metacritic_user_norm'. So let's rename it: 
df.rename(columns={'Metacritic_user_nom':'Metacritic_user_norm'}, inplace=True)

RT_lst = df['RT_norm'] >= 4.
IMDB_lst = df['IMDB_norm'] >=4. 
rankings_lst = ['Fandango_Stars',
                'RT_user_norm',
                'RT_norm',
                'IMDB_norm',
                'Metacritic_user_norm',
                'Metacritic_norm']
# compute the Pearson correlation coefficients and build a full correlation matrix 
def my_heatmap(df):    
    import seaborn as sns    
    fig, axes = plt.subplots()
    sns.heatmap(df, annot=True)
    plt.show()
    plt.close()
    
my_heatmap(df[rankings_lst].corr(method='pearson'))
my_heatmap(df[RT_lst][rankings_lst].corr(method='pearson'))


# create a feature matrix 'X' by selecting two DataFrame columns
feature_cols = ['RT_user_norm', 'RT_norm', 'Metacritic_user_norm', 'Metacritic_norm']
X = df.loc[:, feature_cols]

# create a response vector 'y' by selecting a Series
y = df['IMDB_norm']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=43)
# Change 'random_state' value to obtain different final results

dim = len(feature_cols)
# Include extra dimension for independent coefficient
dim += 1

# Include extra column 'independent' for independent coefficient
X_train = X_train.assign( independent = pd.Series([1] * len(y_train), index=X_train.index))
X_test = X_test.assign( independent = pd.Series([1] * len(y_train), index=X_test.index))

# Convert panda dataframes to numpy arrays
P_train = X_train.as_matrix(columns=None)
P_test = X_test.as_matrix(columns=None)

q_train = np.array(y_train.values).reshape(-1,1)
q_test = np.array(y_test.values).reshape(-1,1)

# Creating placeholder for TensorFlow
P = tf.placeholder(tf.float32,[None,dim])
q = tf.placeholder(tf.float32,[None,1])
T = tf.Variable(tf.ones([dim,1]))

# Adding some extra bias 
bias = tf.Variable(tf.constant(1.0, shape = [dim]))
q_ = tf.add(tf.matmul(P, T),bias)

# Creating an optimizer to optimize the cost fucntion
cost = tf.reduce_mean(tf.square(q_ - q))
learning_rate = 0.0001
training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Variable initilizer
init_op = tf.global_variables_initializer()
cost_history = np.empty(shape=[1],dtype=float)

training_epochs = 50000
with tf.Session() as sess:
    sess.run(init_op)
    cost_history = np.empty(shape=[1], dtype=float)
    t_history = np.empty(shape=[dim, 1], dtype=float)
    for epoch in range(training_epochs):
        sess.run(training_op, feed_dict={P: P_train, q: q_train})
        cost_history = np.append(cost_history, sess.run(cost, feed_dict={P: P_train, q: q_train}))
        t_history = np.append(t_history, sess.run(T, feed_dict={P: P_train, q: q_train}), axis=1)
    q_pred = sess.run(q_, feed_dict={P: P_test})[:, 0]
    mse = tf.reduce_mean(tf.square(q_pred - q_test))
    mse_temp = mse.eval()
    sess.close()

print(mse_temp)
RMSE = math.sqrt(mse_temp)
print(RMSE)

# Observing the training cost throughout iterations
fig, axes = plt.subplots()
plt.plot(range(len(cost_history)), cost_history)
axes.set_xlim(xmin=0.95)
axes.set_ylim(ymin=1.e-2)
axes.set_xscale("log", nonposx='clip')
axes.set_yscale("log", nonposy='clip')
axes.set_ylabel('Training cost')
axes.set_xlabel('Iterations')
axes.set_title('Learning rate = ' + str(learning_rate))
plt.show()
plt.close()


# Predicted vs Actual
predictedDF = X_test.copy(deep=True)
predictedDF.insert(loc=0, column='IMDB_norm_predicted', value=pd.Series(data=q_pred, index=predictedDF.index))
predictedDF.insert(loc=0, column='IMDB_norm_actual', value=q_test)

print('Predicted vs actual rating using LR with TensorFlow')
print(predictedDF[['IMDB_norm_actual', 'IMDB_norm_predicted']].head())
print(predictedDF[['IMDB_norm_actual', 'IMDB_norm_predicted']].tail())

# Let's see how the LR fit with the predicted data points
plt.scatter(q_test, q_pred, color='blue', alpha=0.5)
plt.plot([q_test.min(), q_test.max()], [q_test.min(), q_test.max()], '--', lw=1)
plt.title('Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
