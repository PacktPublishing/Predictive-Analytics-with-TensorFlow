import os
import shutil
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from feature import *

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import svm
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

random.seed(1985)

train = pd.read_csv(os.path.join('input', 'train.csv'))
test = pd.read_csv(os.path.join('input', 'test.csv'))

# Do feature engineering
train, test = create_name_feat(train, test)
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

def train_input_fn():
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    # https://github.com/tensorflow/tensorflow/issues/9505
    continuous_cols = {k: tf.expand_dims(tf.constant(train[k].values), 1)
                       for k in list(train) if k not in ['Survived', 'PassengerId']}

    id_col = {'PassengerId' : tf.constant(train['PassengerId'].values)}

    # Merges the two dictionaries into one.
    feature_cols = continuous_cols.copy()
    feature_cols.update(id_col)
    # Converts the label column into a constant Tensor.
    label = tf.constant(train["Survived"].values)
    # Returns the feature columns and the label.
    return feature_cols, label

def predict_input_fn():
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    # https://github.com/tensorflow/tensorflow/issues/9505
    continuous_cols = {k: tf.expand_dims(tf.constant(test[k].values), 1)
                       for k in list(test) if k not in ['Survived', 'PassengerId']}

    id_col = {'PassengerId' : tf.constant(test['PassengerId'].values)}

    # Merges the two dictionaries into one.
    feature_cols = continuous_cols.copy()
    feature_cols.update(id_col)
    # Converts the label column into a constant Tensor.
    # Returns the feature columns and the label.
    return feature_cols

try:
    shutil.rmtree('svm/')
except OSError:
    pass

svm_model = svm.SVM(example_id_column="PassengerId",
                    feature_columns=[tf.contrib.layers.real_valued_column(k) for k in list(train)
                                     if k not in ['Survived', 'PassengerId']],
                    model_dir="svm/")
svm_model.fit(input_fn=train_input_fn, steps=100)
svm_pred = list(svm_model.predict_classes(input_fn=predict_input_fn))

if TEST:
    target_names = ['Not Survived', 'Survived']
    print("SVM Report")
    print(classification_report(test['Survived'], svm_pred, target_names=target_names))

    print("SVM Confusion Matrix")
    cm = confusion_matrix(test['Survived'], svm_pred)
    df_cm = pd.DataFrame(cm, index=[i for i in ['Not Survived', 'Survived']],
                         columns=[i for i in ['Not Survived', 'Survived']])
    print(df_cm)

# From the classification report we can see that RandomForest has the best overall performance.
# The reason for this may be it works better with categorical features than the other two methods 
# Also since it has implicit feature selection overfitting will also going to reduce.

sol = pd.DataFrame()
sol['PassengerId'] = test['PassengerId']

sol['Survived'] = pd.Series(svm_pred).values
sol.to_csv('submission_svm.csv', index=False)

plt.suptitle("Predicted Survived SVM")
count_plot = sns.countplot(sol.Survived)
count_plot.get_figure().savefig("survived_count_svm_prd.png")

print("Predicted Counts")
print(sol.Survived.value_counts())
