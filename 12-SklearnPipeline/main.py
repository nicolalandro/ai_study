import pickle

import pandas as pd
from sklearn.cluster import Birch, KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

examples = []
truths = []

dataset = pd.read_csv('makeup_dataset.csv', sep='|')
for index, row in dataset.iterrows():
    examples.append(str(row[1]) + ' ' + str(row[2]))
    truths.append(str(row[4]))

train_examples, test_examples, train_truths, test_truths = train_test_split(examples, truths, test_size=0.33)

dictionary = CountVectorizer().fit(train_examples)
transformed_train_example = dictionary.transform(train_examples)
decision_tree = ExtraTreesClassifier(n_estimators=100, n_jobs=12, bootstrap=False, min_samples_split=2, random_state=0)
decision_tree.fit(transformed_train_example, train_truths)

transform_test_example = dictionary.transform(test_examples)
prediction = decision_tree.predict(transform_test_example)

# print(prediction)

text_clf_extra_tree = Pipeline([('vect', CountVectorizer()),
                                ('clf-extra-tree',
                                 ExtraTreesClassifier(n_estimators=100, n_jobs=12, bootstrap=False, min_samples_split=2,
                                                      random_state=0))
                                ])
text_clf_extra_tree.fit(train_examples, train_truths)
text_clf_prediction = text_clf_extra_tree.predict(test_examples)
# print(text_clf_prediction)
print("Extra tree with count vectorizer only perecision: ", accuracy_score(test_truths, text_clf_prediction))

text_clf_with_tfidf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf-extra-tree',
     ExtraTreesClassifier(n_estimators=100, n_jobs=12, bootstrap=False, min_samples_split=2,
                          random_state=0))
])
text_clf_with_tfidf.fit(train_examples, train_truths)
text_clf_with_tfidf_prediction = text_clf_with_tfidf.predict(test_examples)
print("Extra tree with tfidf perecision: ", accuracy_score(test_truths, text_clf_with_tfidf_prediction))

s = pickle.dumps(text_clf_with_tfidf)
clf_restored = pickle.loads(s)
clf_restored.fit(train_examples, train_truths)
clf_restored_prediction = clf_restored.predict(test_examples)
# print(clf_restored_prediction)

clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', DecisionTreeClassifier())
])
clf.fit(train_examples, train_truths)
clf_prediction = clf.predict(test_examples)
print("decision tree perecision: ", accuracy_score(test_truths, clf_prediction))

clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('automatic', Birch()),
    ('clf', ExtraTreesClassifier())
])
clf.fit(train_examples, train_truths)
clf_prediction = clf.predict(test_examples)
print("Birch + ExtraTree: ", accuracy_score(test_truths, clf_prediction))

clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('automatic', KMeans(n_clusters=80)),
    ('clf', ExtraTreesClassifier())
])
clf.fit(train_examples, train_truths)
clf_prediction = clf.predict(test_examples)
print("KMeans + ExtraTree: ", accuracy_score(test_truths, clf_prediction))