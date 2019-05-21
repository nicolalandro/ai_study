import shap

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

shap.initjs()

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

decision_tree_predictions = decision_tree.predict(x_test)
print("decision tree perecision: ", accuracy_score(y_test, decision_tree_predictions))

explainer = shap.TreeExplainer(decision_tree)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test)

