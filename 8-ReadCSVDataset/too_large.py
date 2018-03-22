import pandas as pd
import numpy as np

# if csv is too large to load in memory you can divide it in chunk
file_name = 'PantheonProject:HistoricalPopularityIndex.csv'
column_name = names=['article_id', 'full_name', 'sex', 'birth_year', 'city', 'state', 'country', 'continent', 'latitude', 'longitude', 'occupation', 'industry', 'domain', 'article_lenguages', 'page_reviews', 'avarage_views', 'historical_popularity_index']
# 15000 equivale a 1.5 Mb
chunksize = 1024
counter = 0
for chunk in pd.read_csv(file_name, chunksize=chunksize, names=column_name):
    print('<------------ CHUNK ------------>', counter)
    counter += 1
    for index, row in chunk.iterrows():
        print(row['full_name'])
