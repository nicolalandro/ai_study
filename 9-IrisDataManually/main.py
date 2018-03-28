import pandas as pd
import tensorflow as tf

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train = pd.read_csv('iris_training.csv', names=CSV_COLUMN_NAMES)
test = pd.read_csv('iris_test.csv', names=CSV_COLUMN_NAMES)

for index, row in train.iterrows():
    try:
        numeric_species = int(row['Species'])
        print SPECIES[numeric_species] 
    except:
        print row['Species']
