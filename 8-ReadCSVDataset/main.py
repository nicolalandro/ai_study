import pandas as pd
import numpy as np

df = pd.read_csv('PantheonProject:HistoricalPopularityIndex.csv',
        names=['article_id', 'full_name', 'sex', 'birth_year', 'city', 'state', 'country', 'continent', 'latitude', 'longitude', 'occupation', 'industry', 'domain', 'article_lenguages', 'page_reviews', 'avarage_views', 'historical_popularity_index']
    )
for index, row in df.iterrows():
    print(row['full_name'])
