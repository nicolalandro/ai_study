from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
decision_tree_predictions = decision_tree.predict(x_test)

from sklearn.metrics import accuracy_score
print("decision tree perecision: ", accuracy_score(y_test, decision_tree_predictions))

# save tree in readable format
from sklearn.tree import export_graphviz
export_graphviz(decision_tree, out_file='iris_decision_tree.dot')

# save tree for future use
from sklearn.externals import joblib
joblib.dump(decision_tree, 'iris_decision_tree.pkl')
# restore tree 
restored_decision_tree = joblib.load('iris_decision_tree.pkl')
