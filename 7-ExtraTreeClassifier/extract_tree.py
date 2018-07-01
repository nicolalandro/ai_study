import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz

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

trees = text_clf_extra_tree.named_steps['clf-extra-tree'].estimators_
size = str(len(trees))
for i, tree in enumerate(trees):
    print(str(i+1) + '/' + size)
    name = 'trees/' + str(i) + '-tree.dot'
    export_graphviz(tree, out_file=name)
