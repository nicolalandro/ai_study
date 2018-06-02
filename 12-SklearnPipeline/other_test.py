import pandas as pd
from sklearn.cluster import Birch, KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

examples = []
truths = []

dataset = pd.read_csv('makeup_dataset.csv', sep='|')
for index, row in dataset.iterrows():
    examples.append(str(row[1]) + ' ' + str(row[2]))
    truths.append(str(row[4]))

train_examples, test_examples, train_truths, test_truths = train_test_split(examples, truths, test_size=0.33)

clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('automatic', KMeans(n_clusters=80)),
    ('clf', ExtraTreesClassifier())
])
clf.fit(train_examples, train_truths)
clf_prediction = clf.predict(test_examples)
print("KMeans + ExtraTree: ", accuracy_score(test_truths, clf_prediction))
