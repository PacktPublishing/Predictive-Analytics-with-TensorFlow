import os
import shutil
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from feature import *

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib import learn
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

random.seed(1985)

def build_lr_estimator(model_dir, feature_count):
    return estimator.SKCompat(learn.LinearClassifier(
        feature_columns=[tf.contrib.layers.real_valued_column("", dimension=feature_count)],
        n_classes=2, model_dir=model_dir))

train = pd.read_csv(os.path.join('input', 'train.csv'))
test = pd.read_csv(os.path.join('input', 'test.csv'))

# Do feature engineering
train, test = create_name_feat(train, test)
print(train.head())
train, test = age_impute(train, test)
train, test = cabin(train, test)
train, test = embarked_impute(train, test)
train, test = fam_size(train, test)
test['Fare'].fillna(train['Fare'].mean(), inplace=True)
train, test = ticket_grouped(train, test)

# It next line is used to create numerical values for string variables. The dummies() fucntion does a one-hot encoding to the string variables 
train, test = dummies(train, test, columns=['Pclass', 'Sex', 'Embarked', 'Ticket_Letr', 'Cabin_Letter', 'Name_Title', 'Fam_Size'])

TEST = True
if TEST:
    train, test = train_test_split(train, test_size=0.2, random_state=10)
    train = train.sort_values('PassengerId')
    test = test.sort_values('PassengerId')

# Convert PassengerId to string. This is needed for SVM
train['PassengerId'] = train['PassengerId'].astype(str)
test['PassengerId'] = test['PassengerId'].astype(str)

x_train = train.iloc[:, 1:]
x_test = test.iloc[:, 1:]

x_train = np.array(x_train.iloc[:, 1:], dtype='float32')
if TEST:
    x_test = np.array(x_test.iloc[:, 1:], dtype='float32')
else:
    x_test = np.array(x_test, dtype='float32')

y_train = PrepareTarget(train)
feature_count = x_train.shape[1]

print(x_train.shape)
print(x_test.shape)

# Train and evaluate the model.
print("Training...")

try:
    shutil.rmtree('lr/')
except OSError:
    pass

lr = build_lr_estimator('lr/', feature_count)
lr.fit(x_train, y_train, steps=100)
lr_pred = lr.predict(x_test)
#print(lr_pred['probabilities'])
lr_pred = lr_pred['classes']

print(len(lr_pred))

if TEST:
    target_names = ['Not Survived', 'Survived']

    print("Logistic Regression Report")
    print(classification_report(test['Survived'], lr_pred, target_names=target_names))
    print("Logistic Regression Confusion Matrix")
    cm = confusion_matrix(test['Survived'], lr_pred)
    df_cm = pd.DataFrame(cm, index=[i for i in ['Not Survived', 'Survived']],
                         columns=[i for i in ['Not Survived', 'Survived']])
    print(df_cm)

# From the classification report we can see that RandomForest has the best overall performance.
# The reason for this may be it works better with categorical features than the other two methods 
# Also since it has implicit feature selection overfitting will also going to reduce.

sol = pd.DataFrame()
sol['PassengerId'] = test['PassengerId']

sol['Survived'] = pd.Series(lr_pred.reshape(-1)).map({True:1, False:0}).values
sol.to_csv('submission_lr.csv', index=False)

plt.suptitle("Predicted Survived LR")
count_plot = sns.countplot(sol.Survived)
count_plot.get_figure().savefig("survived_count_lr_prd.png")

print("Predicted Counts")
print(sol.Survived.value_counts())
