# https://www.kaggle.com/eliotbarr/text-mining-with-sklearn-keras-mlp-lstm-cnn
import random
import re

import nltk
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from nbsvm import NBSVM

english_stemmer = nltk.stem.SnowballStemmer('english')


def review_to_wordlist(review, remove_stopwords=True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 1. Remove HTML
    review_text = BeautifulSoup(review, "html.parser").get_text()

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

all_data = pd.read_csv(data_file, delimiter=",", skiprows=skip)
print(all_data.shape)
data = all_data[all_data['Reviews'].isnull() == False]
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

# Select best features
# fselect = SelectKBest(chi2, k=10000)
# train_features = fselect.fit_transform(train_features, train["Rating"])
# test_features = fselect.transform(test_features)

model1 = MultinomialNB(alpha=0.001)
model1.fit(train_features, train["Rating"])

model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
model2.fit(train_features, train["Rating"])

model3 = RandomForestClassifier()
model3.fit(train_features, train["Rating"])

model4 = GradientBoostingClassifier()
model4.fit(train_features, train["Rating"])

model5 = NBSVM(C=0.01)
model5.fit(train_features, train["Rating"])

pred_1 = model1.predict(test_features.toarray())
pred_2 = model2.predict(test_features.toarray())
pred_3 = model3.predict(test_features.toarray())
pred_4 = model4.predict(test_features.toarray())
pred_5 = model5.predict(test_features)

print("MultinomialNB accuracy_score: ", accuracy_score(test["Rating"], pred_1))
print("SGDClassifier accuracy_score: ", accuracy_score(test["Rating"], pred_2))
print("RandomForestClassifier accuracy_score: ", accuracy_score(test["Rating"], pred_3))
print("GradientBoostingClassifier accuracy_score: ", accuracy_score(test["Rating"], pred_4))
print("NBSVM accuracy_score: ", accuracy_score(test["Rating"], pred_5))
