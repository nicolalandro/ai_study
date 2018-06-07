import glob
import os

import pandas as pd
from PIL import Image

base_dir = '../flowers'
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
