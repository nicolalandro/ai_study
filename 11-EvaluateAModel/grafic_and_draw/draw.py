import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_recall_curve, classification_report, \
    confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

pd.set_option('max.columns', None)

df = pd.read_csv('Pokemon.csv', low_memory=False)
# print(df.info())
# print(df.head())
print('Legendary:', str(len(df[df['Legendary'] == True]) / len(df) * 100) + '%')

plt.title('Count Plot')
plt.xticks(rotation=45)
sns_plot = sns.countplot(df['Type 1'])
sns_plot.figure.savefig("figure.png")
