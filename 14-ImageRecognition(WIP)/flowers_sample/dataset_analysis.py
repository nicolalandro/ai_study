import glob
import os

import numpy as np
import pandas as pd
from PIL import Image
from keras import Sequential, optimizers
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

base_dir = '/home/mint/.kaggle/datasets/alxmamaev/flowers-recognition/flowers'
classes_folders = os.listdir(base_dir)

dict_s = {
    'file_path': [],
    'class_name': [],
    'h': [],
    'w': [],
    'size': [],  # in byte
    'color_channels': []
}

for class_name in classes_folders:
    for image_name in glob.glob(os.path.join(base_dir, class_name, '*.jpg')):
        path = os.path.join(image_name)
        im = Image.open(path)
        width, height = im.size
        dict_s['file_path'].append(image_name)
        dict_s['class_name'].append(class_name)
        dict_s['h'].append(height)
        dict_s['w'].append(width)
        dict_s['size'].append(os.stat(path).st_size)
        dict_s['color_channels'].append(im.mode)

pd_value = pd.DataFrame.from_dict(dict_s)

print(pd_value[['class_name']].groupby('class_name').size().reset_index(name='counts'))
print(pd_value[['h', 'w', 'size']].agg(['mean', 'median', 'max', 'min']))

# salmples_file_path = pd_value['file_path']
# truths = pd_value['class_name']
# x_train, x_test, y_train, y_test = train_test_split(salmples_file_path, truths, test_size=0.33)

img_width, img_height = 150, 150
batch_size = 32

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

x = load_img(pd_value['file_path'][200], target_size=(img_width, img_height))
x = img_to_array(x)
x = np.expand_dims(x, axis=0)
result = model.predict(x)
print(result.argmax(1))

