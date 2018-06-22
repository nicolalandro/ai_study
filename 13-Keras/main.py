import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from keras_model import KerasModel
from sklearn.preprocessing import LabelEncoder

examples = []
truths = []

dataset = pd.read_csv('../12-SklearnPipeline/makeup_dataset.csv', sep='|')
for index, row in dataset.iterrows():
    examples.append(str(row[1]) + ' ' + str(row[2]))
    truths.append(str(row[4]))

train_examples, test_examples, train_truths, test_truths = train_test_split(examples, truths, test_size=0.33)

text_clf_extra_tree = Pipeline([('vect', CountVectorizer()),
                                ('clf-extra-tree',
                                 ExtraTreesClassifier(n_estimators=100, n_jobs=12, bootstrap=False, min_samples_split=2,
                                                      random_state=0))
                                ])
text_clf_extra_tree.fit(train_examples, train_truths)
text_clf_prediction = text_clf_extra_tree.predict(test_examples)

print("Extra tree with count vectorizer perecision: ", accuracy_score(test_truths, text_clf_prediction))

le = LabelEncoder()
le.fit(train_truths)
train_truths = le.transform(train_truths)

clf = Pipeline([('vect', CountVectorizer(max_features=4000)),
                ('clf-keras', KerasModel())
                ])
clf.fit(train_examples, train_truths)
pred = clf.predict(test_examples).argmax(1)

print("Extra tree with count vectorizer perecision: ", accuracy_score(le.transform(test_truths), pred))

print(le.inverse_transform(clf.predict(["rossetto rosso"]).argmax(1)[0]))
