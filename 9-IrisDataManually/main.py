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

# keras Area
import keras
from keras.models import Sequential

model = Sequential()
from keras.layers import Dense

model.add(Dense(units=8, activation='relu', input_dim=4))
model.add(Dense(units=16, activation='relu', input_dim=4))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(train_examples, train_truths, epochs=20, batch_size=32)

loss_and_metrics = model.evaluate(test_examples, test_truths, batch_size=128)
print("loss and metrics: ", loss_and_metrics)

prediction = model.predict(test_examples, batch_size=128)
prediction = prediction.argmax(1)

from sklearn.metrics import accuracy_score

print("result with neural network: ", accuracy_score(test_truths, prediction))
