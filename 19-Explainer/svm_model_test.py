import shap

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

model = svm.SVC(kernel='rbf', probability=True, gamma='scale')
model.fit(x_train, y_train)

model_predictions = model.predict(x_test)
print("accuracy score: ", accuracy_score(y_test, model_predictions))

explainer = shap.KernelExplainer(model.predict_proba, x_test, link="logit")
shap_values = explainer.shap_values(x_test, nsamples=100)
shap.summary_plot(shap_values, x_test)

