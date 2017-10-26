import pandas as pd
import numpy as np
from run import prediction
import tensorflow as tf
import time
import os

np.random.seed(12345)

def top_k_movies(users,ratings_df,k):
    """
    Returns top k movies for respective user
    INPUTS :
        users      : list of numbers or number , list of user ids
        ratings_df : rating dataframe, store all users rating for respective movies
        k          : natural number
    OUTPUT:
        Dictionary conatining user id as key and list of top k movies for that user as value
    """
    # Extract unseen movies
    dicts={}
    if type(users) is not list:
        users=[users]
    for user in users:
        rated_movies=ratings_df[ratings_df['user']==user].drop(['st', 'user'], axis=1)
        rated_movie=list(rated_movies['item'].values)
        total_movies=list(ratings_df.item.unique())
        unseen_movies=list(set(total_movies) - set(rated_movie))
        rated_list = []
        rated_list=prediction(np.full(len(unseen_movies),user),np.array(unseen_movies))
        useen_movies_df=pd.DataFrame({'item': unseen_movies,'rate':rated_list})
        top_k=list(useen_movies_df.sort_values(['rate','item'], ascending=[0, 0])['item'].head(k).values)
        dicts.update({user:top_k})
    result=pd.DataFrame(dicts)
    result.to_csv("user_top_k.csv")
    return dicts


def user_rating(users,movies):
    """
    Returns user rating for respective user
    INPUTS :
        users      : list of numbers or number, list of user ids or just user id
        movies : list of numbers or number, list of movie ids or just movie id
    OUTPUT:
        list of predicted movies
    """
    if type(users) is not list:
        users=np.array([users])
    if type(movies) is not list:
        movies=np.array([movies])
    return prediction(users,movies)

def top_k_similar_items(movies,ratings_df,k,TRAINED=False):
    """
    Returns k similar movies for respective movie
    INPUTS :
        movies : list of numbers or number, list of movie ids
        ratings_df : rating dataframe, store all users rating for respective movies
        k          : natural number
        TRAINED    : TRUE or FALSE, weather use trained user vs movie table or untrained
    OUTPUT:
        list of k similar movies for respected movie
    """
    if TRAINED:
        df=pd.read_pickle("user_item_table_train.pkl")
    else:
        df=pd.read_pickle("user_item_table.pkl")

    corr_matrix=item_item_correlation(df,TRAINED)
    if type(movies) is not list:
        return corr_matrix[movies].sort_values(ascending=False).drop(movies).index.values[0:k]
    else:
        dict={}
        for movie in movies:
            dict.update({movie:corr_matrix[movie].sort_values(ascending=False).drop(movie).index.values[0:k]})
        pd.DataFrame(dict).to_csv("movie_top_k.csv")
        return dict

def user_similarity(user_1,user_2,ratings_df,TRAINED=False):
    """
    Return the similarity between two users
    INPUTS :
        user_1,user_2 : number, respective user ids
        ratings_df : rating dataframe, store all users rating for respective movies
        TRAINED    : TRUE or FALSE, weather use trained user vs movie table or untrained
    OUTPUT:
        Pearson cofficient between users [value in between -1 to 1]
    """
    corr_matrix=user_user_pearson_corr(ratings_df,TRAINED)

    return corr_matrix[user_1][user_2]


def item_item_correlation(df,TRAINED):
    if TRAINED:
        if os.path.isfile("model/item_item_corr_train.pkl"):
            df_corr=pd.read_pickle("item_item_corr_train.pkl")
        else:
            df_corr=df.corr()
            df_corr.to_pickle("item_item_corr_train.pkl")
    else:
        if os.path.isfile("model/item_item_corr.pkl"):
            df_corr=pd.read_pickle("item_item_corr.pkl")
        else:
            df_corr=df.corr()
            df_corr.to_pickle("item_item_corr.pkl")
    return df_corr


def user_user_pearson_corr(ratings_df,TRAINED):
    if TRAINED:
        if os.path.isfile("model/user_user_corr_train.pkl"):
            df_corr=pd.read_pickle("user_user_corr_train.pkl")
        else:
            df =pd.read_pickle("user_item_table_train.pkl")
            df=df.T
            df_corr=df.corr()
            df_corr.to_pickle("user_user_corr_train.pkl")
    else:
        if os.path.isfile("model/user_user_corr.pkl"):
            df_corr=pd.read_pickle("user_user_corr.pkl")
        else:
            df = pd.read_pickle("user_item_table.pkl")
            df=df.T
            df_corr=df.corr()
            df_corr.to_pickle("user_user_corr.pkl")
    return df_corr
