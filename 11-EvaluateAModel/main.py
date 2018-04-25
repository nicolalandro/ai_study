from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn
import matplotlib.pyplot

iris = load_iris()
x = iris.data
y = iris.target
sns_plot = seaborn.countplot(y)
sns_plot.figure.savefig("rating.png")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
decision_tree_predictions = decision_tree.predict(x_test)


cnf_matrix = confusion_matrix(y_test, decision_tree_predictions)
sns_plot = seaborn.heatmap(cnf_matrix, annot=True, fmt="d", center=0)
sns_plot.figure.savefig("cnf_matrix.png")
print("decision tree perecision: ", accuracy_score(y_test, decision_tree_predictions))
