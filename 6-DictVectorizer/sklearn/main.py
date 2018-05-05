from sklearn.feature_extraction import DictVectorizer

v = DictVectorizer(sparse=False)
D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
X = v.fit_transform(D)
print(X)
print(v.inverse_transform(X))

print(v.transform({'foo': 4, 'unseen_feature': 3}))

measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
]

vec = DictVectorizer()

print("\n\n", vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())
