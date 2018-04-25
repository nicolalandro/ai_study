# https://www.kaggle.com/eliotbarr/text-mining-with-sklearn-keras-mlp-lstm-cnn
import random
import re

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import seaborn as sns

english_stemmer = nltk.stem.SnowballStemmer('english')


def review_to_wordlist(review, remove_stopwords=True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()

    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review)

    # 3. Convert words to lower case and split them
    words = review_text.lower().split()

    # 4. Optionally remove stop words (True by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    b = []
    stemmer = english_stemmer  # PorterStemmer()
    for word in words:
        b.append(stemmer.stem(word))

    # 5. Return a list of words
    return (b)


data_file = 'Amazon_Unlocked_Mobile.csv'
# We import only 20000 lines of our total data in order to run the notebook faster
n = 413000
s = 20000
skip = sorted(random.sample(range(1, n), n - s))

data = pd.read_csv(data_file, delimiter=",", skiprows=skip)
print(data.shape)
data = data[data['Reviews'].isnull() == False]
train, test = train_test_split(data, test_size=0.3)

sns_plot = sns.countplot(data['Rating'])
sns_plot.figure.savefig("rating.png")

clean_train_reviews = []
for review in train['Reviews']:
    clean_train_reviews.append(" ".join(review_to_wordlist(review)))

clean_test_reviews = []
for review in test['Reviews']:
    clean_test_reviews.append(" ".join(review_to_wordlist(review)))

vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, max_features=200000, ngram_range=(1, 4),
                             sublinear_tf=True)

vectorizer = vectorizer.fit(clean_train_reviews)
train_features = vectorizer.transform(clean_train_reviews)

test_features = vectorizer.transform(clean_test_reviews)
