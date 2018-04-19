import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_csv = pd.read_csv('iris_training.csv', names=CSV_COLUMN_NAMES)
train_examples = train_csv.drop(['Species'], axis=1).values
train_truths = train_csv['Species'].get_values()

test_csv = pd.read_csv('iris_test.csv', names=CSV_COLUMN_NAMES)
test_examples = test_csv.drop(['Species'], axis=1).values
test_truths = test_csv['Species'].get_values()

# SKlearn Area
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier()
clf.fit(train_examples, train_truths)
prediction = clf.predict(test_examples)
print("result with decision tree:", accuracy_score(test_truths, prediction))

# Tensorflow Area
import tensorflow as tf 


# for index, row in train_csv.iterrows():
#     try:
#         numeric_species = int(row['Species'])
#         print(SPECIES[numeric_species])
#     except:
#         print(row['Species'])
