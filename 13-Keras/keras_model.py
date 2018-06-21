import keras
from keras import Sequential
from keras.layers import Dense


class KerasModel:
    def __init__(self) -> None:
        super().__init__()
        self.model = Sequential()
        self.model.add(Dense(units=32, activation='relu', input_dim=4000))
        self.model.add(Dense(units=16, activation='relu', input_dim=32))
        self.model.add(Dense(units=23, activation='softmax'))
        self.model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                           optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

    def fit(self, x, y):
        self.model.fit(x, y, epochs=40, batch_size=32)

    def predict(self, x):
        return self.model.predict(x, batch_size=128)
