x = [
    'geotel g1-5,0" 3g smartphone, android 7.0 quad core 2gb+16gb, 7500mah funzione power bank, fotocamera',
    'samsung galaxy a8 (2018) smartphone, black, 32gb espandibili, dual sim',
    'canon pixma pro-100s stampante multifunzione inkjet wi-fi 4800 x 2400 dpi, nero',
    'epson xp-342 inkjet stampante multifunzione con cartucce di inchiostro separate, nero'
]

from sklearn.feature_extraction.text import CountVectorizer

dictionary = ['samsung', 'canon', 'pixma', 'multifunzione', 'smartphone', 'android', 'inkjet', 'inchiostro', 'cartucce',
              'epson', 'geotel']
cv = CountVectorizer()
cv.fit(dictionary)
transformed_x = cv.transform(x)
print('transformed_x:', transformed_x.toarray())

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(transformed_x)
print('kmeans label:',kmeans.labels_)
# [0 0 1 1]

x_test = ['samsung s9', 'canon inkjet']
transformed_x_test = cv.transform(x_test)
kmeans_prediction = kmeans.predict(transformed_x_test)

from sklearn.metrics import accuracy_score
y_test = [0, 1]

print(
    "k means perecision:",
    accuracy_score(y_test, kmeans_prediction)
)
