import pandas as pd
import numpy as np

#########################################################################################################
'''
Feature Engineering
'''
def create_name_feat(train, test):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i['Name']
    return train, test

# There are 177 null values for Age, and those ones have a 10% lower survival rate than the non-nulls. 
# Before imputing values for the nulls, we are including an Age_null flag just to make 
# sure we can account for this characteristic of the data.

def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train, test

# We combine the SibSp and Parch columns to create get family size and break it to three levels
def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0, 'One',
                                 np.where((i['SibSp']+i['Parch']) <= 3, 'Small', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test

# We are using the Ticket column to create Ticket_Letr, which indicates the first letter 
# of each ticket # and Ticket_Len, which indicates the length of the Ticket field
def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Letr'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Letr'] = i['Ticket_Letr'].apply(lambda x: str(x))
        i['Ticket_Letr'] = np.where((i['Ticket_Letr']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']),
                                    i['Ticket_Letr'],
                                    np.where((i['Ticket_Letr']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                             'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test

# Extract the first letter of the Cabin column
def cabin(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test

# We fill the null values in the Embarked column with the most commonly occuring value, which is 'S'
def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test

# Convert our categorical columns into dummy variables
def dummies(train, test,
            columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Letr', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix=column)[good_cols]), axis=1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix=column)[good_cols]), axis=1)
        del train[column]
        del test[column]
    return train, test

def PrepareTarget(data):
    return np.array(data.Survived, dtype='int8').reshape(-1, 1)

