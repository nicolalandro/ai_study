import os

import matplotlib.pyplot as plt

base_dir = '../flowers'
classes_folders = os.listdir(base_dir)
print('classes folders: ', len(classes_folders))
files_name_for_class = []
counter = []
for idx, class_name in enumerate(classes_folders):
    path = os.path.join(base_dir, class_name)
    files_name = os.listdir(path)
    files_name_for_class.append(files_name)
    counter.append(len(files_name))
    print(classes_folders[idx], len(files_name))

fig = plt.bar(classes_folders, counter)
plt.savefig("0-daset_number.png")
plt.close()
