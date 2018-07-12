from keras import Sequential, optimizers
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/home/mint/.kaggle/datasets/alxmamaev/flowers-recognition/flowers'
img_width, img_height = 150, 150
batch_size = 16

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 2, 2, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.0004),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    samples_per_epoch=1000,
    epochs=20)

model.save('model.h5')
model.save_weights('weights.h5')