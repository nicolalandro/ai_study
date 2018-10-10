from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, LSTM, Dropout, TimeDistributed, Activation, Dense
from data_helper import load_data
from keras_batch_generator import KerasBatchGenerator

ai_path = './ai'
hidden_size = 500
num_steps = 30
batch_size = 20
num_epochs = 1 #50

train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)

model = Sequential()
model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
checkpointer = ModelCheckpoint(filepath=ai_path + '/model-{epoch:02d}.hdf5', verbose=1)

model.fit_generator(train_data_generator.generate(), len(train_data) // (batch_size * num_steps), num_epochs,
                    validation_data=valid_data_generator.generate(),
                    validation_steps=len(valid_data) // (batch_size * num_steps), callbacks=[checkpointer])
