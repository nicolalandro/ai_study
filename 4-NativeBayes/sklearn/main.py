from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

from sklearn.naive_bayes import GaussianNB
native_bayes = GaussianNB()
native_bayes.fit(x_train, y_train)
native_bayes_predictions = native_bayes.predict(x_test)

from sklearn.metrics import accuracy_score
print("native bayes perecision: ", accuracy_score(y_test, native_bayes_predictions))
