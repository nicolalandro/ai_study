import unittest

import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense


class TestKerasKoans(unittest.TestCase):

    def test_mlp(self):
        train = np.array([
            [0, 0, 1, 5, 2, 4],
            [30, 1, 76, 2, 1, 2]
        ])

        truth = np.array([
            [0, 1],
            [1, 0]
        ])

        model = Sequential()
        model.add(Dense(units=32, activation='relu', input_dim=len(train[0])))
        model.add(Dense(units=16, activation='relu', input_dim=32))
        model.add(Dense(units=len(truth[0]), activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
        model.fit(train, truth, epochs=1, batch_size=2, verbose=0)

        prediction = model.predict(train)

        self.assertEqual((2, 2), prediction.shape)


if __name__ == '__main__':
    unittest.main()
