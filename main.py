import numpy as np
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

df = pd.merge(df,movie_titles,on="item_id")
print(df.shape)
