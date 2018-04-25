# https://www.kaggle.com/eliotbarr/text-mining-with-sklearn-keras-mlp-lstm-cnn
import itertools
import random
import re

import nltk
import pandas as pd
import numpy as np
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from nbsvm import NBSVM
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import cm

plt.style.use('ggplot')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def f1_class(pred, truth, class_val):
    n = len(truth)

    truth_class = 0
    pred_class = 0
    tp = 0

    for ii in range(0, n):
        if truth[ii] == class_val:
            truth_class += 1
            if truth[ii] == pred[ii]:
                tp += 1
                pred_class += 1
                continue;
        if pred[ii] == class_val:
            pred_class += 1

    precision = tp / float(pred_class)
    recall = tp / float(truth_class)

    return (2.0 * precision * recall) / (precision + recall)


def semeval_senti_f1(pred, truth, pos=2, neg=0):
    f1_pos = f1_class(pred, truth, pos)
    f1_neg = f1_class(pred, truth, neg)

    return (f1_pos + f1_neg) / 2.0;


def main(train_file, test_file, ngram=(1, 3)):
    print('loading...')
    train = pd.read_csv(train_file, delimiter='\t', encoding='utf-8', header=0,
                        names=['text', 'label'])

    # to shuffle:
    # train.iloc[np.random.permutation(len(df))]

    test = pd.read_csv(test_file, delimiter='\t', encoding='utf-8', header=0,
                       names=['text', 'label'])

    print('vectorizing...')
    vect = CountVectorizer()
    classifier = NBSVM()

    # create pipeline
    clf = Pipeline([('vect', vect), ('nbsvm', classifier)])
    params = {
        'vect__token_pattern': r"\S+",
        'vect__ngram_range': ngram,
        'vect__binary': True
    }
    clf.set_params(**params)

    # X_train = vect.fit_transform(train['text'])
    # X_test = vect.transform(test['text'])

    print('fitting...')
    clf.fit(train['text'], train['label'])

    print('classifying...')
    pred = clf.predict(test['text'])

    print('testing...')
    acc = accuracy_score(test['label'], pred)
    f1 = semeval_senti_f1(pred, test['label'])
    print('NBSVM: acc=%f, f1=%f' % (acc, f1))


english_stemmer = nltk.stem.SnowballStemmer('english')


def review_to_wordlist(review, remove_stopwords=True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 1. Remove HTML
    review_text = BeautifulSoup(review, "html.parser").get_text()

    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review)

    # 3. Convert words to lower case and split them
    words = review_text.lower().split()

    # 4. Optionally remove stop words (True by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    b = []
    stemmer = english_stemmer  # PorterStemmer()
    for word in words:
        b.append(stemmer.stem(word))

    # 5. Return a list of words
    return (b)


data_file = 'Amazon_Unlocked_Mobile.csv'
# We import only 20000 lines of our total data in order to run the notebook faster
n = 413000
s = 20000
skip = sorted(random.sample(range(1, n), n - s))

all_data = pd.read_csv(data_file, delimiter=",", skiprows=skip)
print(all_data.shape)
data = all_data[all_data['Reviews'].isnull() == False]
train, test = train_test_split(data, test_size=0.3)

sns_plot = sns.countplot(data['Rating'])
sns_plot.figure.savefig("rating.png")

clean_train_reviews = []
for review in train['Reviews']:
    clean_train_reviews.append(" ".join(review_to_wordlist(review)))

clean_test_reviews = []
for review in test['Reviews']:
    clean_test_reviews.append(" ".join(review_to_wordlist(review)))

vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, max_features=200000, ngram_range=(1, 4),
                             sublinear_tf=True)

vectorizer = vectorizer.fit(clean_train_reviews)
train_features = vectorizer.transform(clean_train_reviews)

test_features = vectorizer.transform(clean_test_reviews)

# Select best features
# fselect = SelectKBest(chi2, k=10000)
# train_features = fselect.fit_transform(train_features, train["Rating"])
# test_features = fselect.transform(test_features)

model1 = MultinomialNB(alpha=0.001)
model1.fit(train_features, train["Rating"])

model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
model2.fit(train_features, train["Rating"])

model3 = RandomForestClassifier()
model3.fit(train_features, train["Rating"])

model4 = GradientBoostingClassifier()
model4.fit(train_features, train["Rating"])

model5 = NBSVM(C=0.01)
model5.fit(train_features, train["Rating"])

pred_1 = model1.predict(test_features.toarray())
pred_2 = model2.predict(test_features.toarray())
pred_3 = model3.predict(test_features.toarray())
pred_4 = model4.predict(test_features.toarray())
pred_5 = model5.predict(test_features)

print("MultinomialNB accuracy_score: ", accuracy_score(test["Rating"], pred_1))
print(classification_report(test['Rating'], pred_1, target_names=['1', '2', '3', '4', '5']))
cnf_matrix = confusion_matrix(test['Rating'], pred_1)
plot_confusion_matrix(cnf_matrix, classes=['1', '2', '3', '4', '5'],
                      title='Confusion matrix, without normalization')

print("SGDClassifier accuracy_score: ", accuracy_score(test["Rating"], pred_2))
print(classification_report(test['Rating'], pred_2, target_names=['1', '2', '3', '4', '5']))

print("RandomForestClassifier accuracy_score: ", accuracy_score(test["Rating"], pred_3))
print(classification_report(test['Rating'], pred_3, target_names=['1', '2', '3', '4', '5']))

print("GradientBoostingClassifier accuracy_score: ", accuracy_score(test["Rating"], pred_4))
print(classification_report(test['Rating'], pred_4, target_names=['1', '2', '3', '4', '5']))

print("NBSVM accuracy_score: ", accuracy_score(test["Rating"], pred_5))
print(classification_report(test['Rating'], pred_5, target_names=['1', '2', '3', '4', '5']))

# https://www.kaggle.com/eliotbarr/text-mining-with-sklearn-keras-mlp-lstm-cnn
# Non fatto da:
# Deep Learning MLP