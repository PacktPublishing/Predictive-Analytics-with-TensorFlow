import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shutil
import tensorflow as tf
import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.learn.python.learn.estimators import svm
from tensorflow.contrib import learn


############################################################################################
''' 
Problem

In this dataset we have a subset of titanic population and we are asked to build 
a predictive model to say whether a given passenger survived or not in the Titanic disaster. 
We have 10 input variables such as gender, age, and price of fare. 
This is going to be a binary classification problem and we are going to solve it using 
RandomForests, Logistic Regression and SVM.

'''
############################################################################################
'''
Exploratory Data Analysis
'''

# First Let's load the data and check what are the features aviable to us
train = pd.read_csv(os.path.join('input', 'train.csv'))
test = pd.read_csv(os.path.join('input', 'test.csv'))

print("Information about the data")
print(train.dtypes)
print(train.info())

# We can see there are several missing data in the dataset and we will have to do some 
# imputation for them.

# How many have survived?

print("How many have survived?")
print(train.Survived.value_counts(normalize=True))
count_plot = sns.countplot(train.Survived)
count_plot.get_figure().savefig("survived_count.png")

# What is the relationship between the class and the rate of survival.
# As you may remember from the movie people from higher classes had a better chance of
# surviving. Let's see wheter this is true.

print('Survival for each Pclass')
train.Survived.groupby(train.Pclass).agg(['mean', 'count'])
count_plot = sns.countplot(train['Pclass'], hue=train['Survived'])
count_plot.get_figure().savefig("survived_count_by_class.png")

# Survival and title

train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
print('Title count')
print(train['Name_Title'].value_counts())
print('Survived by title')
print(train['Survived'].groupby(train['Name_Title']).mean())

# People with longer names has a higer probability of survival.
# This happens due to most of the people with longer names are married ladies

train['Name_Len'] = train['Name'].apply(lambda x: len(x))
print('Survived by name length')
print(train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean())


# Women and children had a higher chance to survive since the are the first to evcuave in the shipwreak
print('Survived by sex')
print(train['Survived'].groupby(train['Sex']).mean())

# One important picece of information in the ticket is the first letter.
# Which may indicate some attribute of the ticketholders or their rooms.

print('Survived by Ticket_Letr')
train['Ticket_Letr'] = train['Ticket'].apply(lambda x: str(x)[0])
train.groupby(['Ticket_Letr'])['Survived'].mean()


# There is a clear relationship between Fare and Survived.
print(train['Survived'].groupby(pd.qcut(train['Fare'], 3)).mean())
print("Relationship between class and fare")
print(pd.crosstab(pd.qcut(train['Fare'], 5), columns=train['Pclass']))

# Cabin has the most nulls (almost 700), but we can still extract information from it, 
# like the first letter of each cabin
# We can see that most of the cabin letters are associated with survival rate
train['Cabin_Letter'] = train['Cabin'].apply(lambda x: str(x)[0])
print('Survived by Cabin_Letter')
print(train['Survived'].groupby(train['Cabin_Letter']).mean())

# Seems people embarked from Cherbourg had a 20% higher survival rate than the other embarking locations. 
# This is very likely due to the high presence of upper-class passengers from that location.
print('Survived by Embarked')
print(train['Survived'].groupby(train['Embarked']).mean())
count_plot = sns.countplot(train['Embarked'], hue=train['Pclass'])
count_plot.get_figure().savefig("survived_count_by_embarked.png")
