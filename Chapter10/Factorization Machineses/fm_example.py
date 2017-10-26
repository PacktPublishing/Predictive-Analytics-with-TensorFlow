import pandas as pd
from collections import Counter
import tensorflow as tf
from tffm import TFFMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.python.framework import ops
import warnings
import random
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

# Loading datasets'
buys = open('Input/yoochoose-buys.dat', 'r')
clicks = open('Input/yoochoose-clicks.dat', 'r')

initial_buys_df = pd.read_csv(buys, names=['Session ID', 'Timestamp', 'Item ID', 'Category', 'Quantity'],
                              dtype={'Session ID': 'float32', 'Timestamp': 'str', 'Item ID': 'float32',
                                     'Category': 'str'})

initial_buys_df.set_index('Session ID', inplace=True)

initial_clicks_df = pd.read_csv(clicks, names=['Session ID', 'Timestamp', 'Item ID', 'Category'],
                                dtype={'Category': 'str'})

initial_clicks_df.set_index('Session ID', inplace=True)

# We won't use timestamps in this example

initial_buys_df = initial_buys_df.drop('Timestamp', 1)
initial_clicks_df = initial_clicks_df.drop('Timestamp', 1)

# For illustrative purposes, we will only use a subset of the data: top 10000 buying users,

x = Counter(initial_buys_df.index).most_common(10000)
top_k = dict(x).keys()
initial_buys_df = initial_buys_df[initial_buys_df.index.isin(top_k)]
initial_clicks_df = initial_clicks_df[initial_clicks_df.index.isin(top_k)]

# Create a copy of the index, since we will also apply one-hot encoding on the index

initial_buys_df['_Session ID'] = initial_buys_df.index

# One-hot encode all columns for clicks and buys

transformed_buys = pd.get_dummies(initial_buys_df)
transformed_clicks = pd.get_dummies(initial_clicks_df)

# Aggregate historical data for Items and Categories

filtered_buys = transformed_buys.filter(regex="Item.*|Category.*")
filtered_clicks = transformed_clicks.filter(regex="Item.*|Category.*")

historical_buy_data = filtered_buys.groupby(filtered_buys.index).sum()
historical_buy_data = historical_buy_data.rename(columns=lambda column_name: 'buy history:' + column_name)

historical_click_data = filtered_clicks.groupby(filtered_clicks.index).sum()
historical_click_data = historical_click_data.rename(columns=lambda column_name: 'click history:' + column_name)

# Merge historical data of every user_id

merged1 = pd.merge(transformed_buys, historical_buy_data, left_index=True, right_index=True)
merged2 = pd.merge(merged1, historical_click_data, left_index=True, right_index=True)

# Create the MF model, you can play around with the parameters

model = TFFMRegressor(
    order=2,
    rank=7,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
    n_epochs=100,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)

merged2.drop(['Item ID', '_Session ID', 'click history:Item ID', 'buy history:Item ID'], 1, inplace=True)
X = np.array(merged2)
X = np.nan_to_num(X)
y = np.array(merged2['Quantity'].as_matrix())

# Split data into train, test

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

#Split testing data in half: Full information vs Cold-start

X_te, X_te_cs, y_te, y_te_cs = train_test_split(X_te, y_te, test_size=0.5)
X_te_cs = pd.DataFrame(X_te_cs, columns=merged2.columns)

# What happens if we only have access to categories and no historical click/purchase data?
# Let's delete historical click and purchasing data for the cold_start test set

for column in X_te_cs.columns:
    if ('buy' in column or 'click' in column) and ('Category' not in column):
        X_te_cs[column] = 0

# Compute the mean squared error for both test sets

model.fit(X_tr, y_tr, show_progress=True)
predictions = model.predict(X_te)
cold_start_predictions = model.predict(X_te_cs)
print('MSE: {}'.format(mean_squared_error(y_te, predictions)))
print('Cold-start MSE: {}'.format(mean_squared_error(y_te_cs, predictions)))
model.destroy()

# Fun fact: Dropping the category columns in the training dataset makes the MSE even smaller
# but doing so means that we cannot tackle the cold-start recommendation problem
'''
MSE: 0.4123308369523053
Cold-start MSE: 2.7871912983642093
'''

