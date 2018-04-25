import seaborn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target
sns_plot = seaborn.countplot(y)
sns_plot.figure.savefig("rating.png")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
decision_tree_predictions = decision_tree.predict(x_test)

plt.close()

cnf_matrix = confusion_matrix(y_test, decision_tree_predictions)
sns_plot = seaborn.heatmap(cnf_matrix, annot=True, center=0)
sns_plot.figure.savefig("cnf_matrix.png")
plt.close()


normalized_cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
sns_plot = seaborn.heatmap(normalized_cnf_matrix, annot=True, center=0)
sns_plot.figure.savefig("normalized_cnf_matrix.png")
plt.close()


print("decision tree perecision: ", accuracy_score(y_test, decision_tree_predictions))
print("decision tree recal macro: ", recall_score(y_test, decision_tree_predictions, average='macro'))
print("decision tree recal micro: ", recall_score(y_test, decision_tree_predictions, average='micro'))
print("decision tree recal weighted: ", recall_score(y_test, decision_tree_predictions, average='weighted'))
print("decision tree recal none: ", recall_score(y_test, decision_tree_predictions, average=None))
