import json
import re

from keras.models import model_from_json
from data_helper import load_data
import numpy as np
from sklearn.metrics import accuracy_score

sequence_length = 56


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(sentences, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


vocabulary_inv = json.load(open('./ai/vocaabulary_inv.json'))
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())[:10]
negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())[:10]

texts = positive_examples + negative_examples
texts = [s.strip() for s in texts]
texts = [clean_str(sent) for sent in texts]
texts = [s.split(" ") for s in texts]
texts = pad_sentences(texts)

labels = [[0, 1] for _ in texts]

x = np.array([[vocabulary[word] for word in sentence] for sentence in texts])
y = np.array(labels)

json_file = open('./ai/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./ai/model.h5")
print("Loaded model from disk")

prediction = loaded_model.predict(x)
pred = prediction.argmax(1)
label = y.argmax(1)
print(accuracy_score(pred, label))
