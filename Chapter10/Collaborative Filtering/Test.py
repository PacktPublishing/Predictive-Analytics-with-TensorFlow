import tensorflow as tf
import pandas as pd
import readers
import main
import kmean as km

df=pd.read_pickle("user_item_table_train.pkl")

ratings_df = readers.read_file("Input/ratings.dat", sep="::")

clusters,movies=km.k_mean_clustering(ratings_df=ratings_df,TRAINED=False)
cluster_df=pd.DataFrame({'movies':movies,'clusters':clusters})
cluster_df.head(10)


main.top_k_similar_items(9,ratings_df=ratings_df,k=10,TRAINED=False)
cluster_df[cluster_df['movies']==1721]
cluster_df[cluster_df['movies']==1369]
cluster_df[cluster_df['movies']==164]
cluster_df[cluster_df['movies']==3081]
cluster_df[cluster_df['movies']==732]
cluster_df[cluster_df['movies']==348]
cluster_df[cluster_df['movies']==647]

# Pearson Correlation between User-User. When you run this User Similarity function, on first run it will take time to give output but after that it's response is in real-time.
main.user_similarity(1,345,ratings_df)

# Similarity between two users
#Rating of User - Aspected rating for a user
ratings_df.head()

main.user_rating(0,1192)
main.user_rating(0,660)

# Top K movie recommendation for User
main.top_k_movies([768],ratings_df,10)
main.top_k_movies(1198,ratings_df,10)
