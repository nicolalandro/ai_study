import glob
import os

import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

base_dir = '/home/mint/.kaggle/datasets/alxmamaev/flowers-recognition/flowers'
classes_folders = os.listdir(base_dir)
img_width, img_height = 150, 150

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

model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

x = load_img(pd_value['file_path'][200], target_size=(img_width, img_height))
x = img_to_array(x)
x = np.expand_dims(x, axis=0)
result = model.predict(x)
print(classes_folders[int(result.argmax(1))])
