import glob
import os
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

base_dir = '../flowers'
classes_folders = os.listdir(base_dir)
# print('classes folders: ', len(classes_folders))
files_name_for_class = []
counter = []
for idx, class_name in enumerate(classes_folders):
    path = os.path.join(base_dir, class_name, '*.jpg')
    files_name = glob.glob(path)
    files_name_for_class.append(files_name)
    counter.append(len(files_name))
    # print(classes_folders[idx], len(files_name))

fig = plt.bar(classes_folders, counter)
plt.savefig("0-daset_number.png")
plt.close()

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
print(pd_value)
