import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

ratings_list = [i.strip().split("::") for i in open('Input/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('Input/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('Input/movies.dat', 'r',encoding='latin-1').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
user_df=pd.DataFrame(users_list, columns=['UserID','Gender','Age','Occupation','ZipCode'])

# Change MovieID as numeric values
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
# Change UserID and Age as numeric values
user_df['UserID'] = user_df['UserID'].apply(pd.to_numeric)
user_df['Age'] = user_df['Age'].apply(pd.to_numeric)

print("User table description:")
print(user_df.head())
print(user_df.describe())


print("Movies table description:")
print(movies_df.head())
print(movies_df.describe())

print("Rating table description:")
print(ratings_df.head())
print(ratings_df.describe())

print("Top Five most rated movies:")
print(ratings_df['MovieID'].value_counts().head())

plt.hist(ratings_df.groupby(['MovieID'])['Rating'].mean().sort_values(axis=0, ascending=False))
plt.title("Movie rating Distrbution")
plt.ylabel('Count of movies')
plt.xlabel('Rating');
plt.show()


user_df.Age.plot.hist()
plt.title("Distribution of users (by ages)")
plt.ylabel('Count of users')
plt.xlabel('Age');
plt.show()

movie_ratings = pd.merge(movies_df, ratings_df)
df=pd.merge(movie_ratings,user_df)

most_rated = df.groupby('Title').size().sort_values(ascending=False)[:25]
print("Top 10 most rated movies with title")
print(most_rated.head(10))

movie_stats = df.groupby('Title').agg({'Rating': [np.size, np.mean]})

print("Highest rated moview with minimum 150 ratings")
print(movie_stats.Rating[movie_stats.Rating['size'] > 150].sort_values(['mean'],ascending=[0]).head())


pivoted = df.pivot_table(index=['MovieID', 'Title'],
                           columns=['Gender'],
                           values='Rating',
                           fill_value=0)

print("Gender biasing towards movie rating")
print(pivoted.head())

pivoted['diff'] = pivoted.M - pivoted.F
print(pivoted.head())
