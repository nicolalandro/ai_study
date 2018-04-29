from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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

print(prediction)

text_clf_extra_tree = Pipeline([('vect', CountVectorizer()),
                                ('clf-extra-tree',
                                 ExtraTreesClassifier(n_estimators=100, n_jobs=12, bootstrap=False, min_samples_split=2,
                                                      random_state=0))
                                ])
text_clf_extra_tree.fit(train_examples, train_truths)
text_clf_prediction = text_clf_extra_tree.predict(test_examples)
print(text_clf_prediction)
