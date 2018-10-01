# load json and create model
from keras.models import model_from_json
from data_helper import load_data

x, y, vocabulary, vocabulary_inv = load_data()


json_file = open('./ai/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./ai/model.h5")
print("Loaded model from disk")

prediction = loaded_model.predict(x)
print(prediction)


