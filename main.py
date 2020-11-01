import pandas as pd
import warnings


# ignore warnings #
warnings.filterwarnings("ignore")

# Read Data #
columns_name = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("ml-100k/u.data", sep="\t", names=columns_name)

movie_titles = pd.read_csv("ml-100k/u.item", sep="\|", header=None)
movie_titles = movie_titles[[0, 1]]
movie_titles.columns = ["item_id", "title"]

df = pd.merge(df, movie_titles, on="item_id")

ratings = pd.DataFrame(df.groupby("title").mean()["rating"])
ratings["number of ratings"] = pd.DataFrame(df.groupby("title").count()["rating"])

movieMat = df.pivot_table(index="user_id", columns="title", values="rating")


# prediction Function #

def predict_movie(movie):
    movie_user_rating = movieMat[movie]
    similar_to_movie = movieMat.corrwith(movie_user_rating)

    corr_movie = pd.DataFrame(similar_to_movie, columns=["correlation"])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings["number of ratings"])
    predictions = corr_movie[corr_movie["number of ratings"] > 100].sort_values("correlation", ascending=False)

    return predictions

predictions = predict_movie("Titanic (1997)")
print(predictions.head())