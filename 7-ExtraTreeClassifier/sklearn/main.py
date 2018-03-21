from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier

x = ["pattern 1 ha testo", "pattern 2 pure"]
y = ["classe1", "classe2"]

count_vect = CountVectorizer()
# patterns = count_vect.fit_transform(offer_text_list)
dictionary = count_vect.fit(x) # cosi posso salvare il dictionari per ricaricarlo
transformed_x = dictionary.transform(x)

clf = ExtraTreesClassifier(n_estimators=100, n_jobs=12, bootstrap=False, min_samples_split=2, random_state=0)
clf.fit(transformed_x, y)

predicted = clf.predict(transformed_x)
probability = clf.predict_proba(transformed_x)

print("predicted: ", predicted)
print("prob: ", probability)
