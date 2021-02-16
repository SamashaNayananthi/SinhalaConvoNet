import os
import shutil
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy import ndimage

def process(img):
    conv = img.convert("L")
    inv = ImageOps.invert(conv)

    bounding_box = inv.getbbox()
    padded_box = tuple(map(lambda i, j: i + j, bounding_box, (-5, -5, 5, 5)))
    cropped = inv.crop(padded_box)

    # pad = ImageOps.expand(inv, 2)
    thick = cropped.filter(ImageFilter.MaxFilter(5))

    ratio = 48.0 / max(thick.size)
    new_size = tuple([int(round(x * ratio)) for x in thick.size])
    res = thick.resize(new_size, Image.LANCZOS)

    arr = np.asarray(res)
    com = ndimage.measurements.center_of_mass(arr)
    result = Image.new("L", (64, 64))
    box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))
    result.paste(res, box)
    return result

# train_root = 'D:/Final Year/Final Year Project/dataset1/train'
# test_root = 'D:/Final Year/Final Year Project/dataset1/test/'
# train_target = 'D:/Final Year/Final Year Project/dataset-processed1/train'
# test_target = 'D:/Final Year/Final Year Project/dataset-processed1/test/'

train_root = 'C:/Users/User/PycharmProjects/FYP1/data/dataset/train'
test_root = 'C:/Users/User/PycharmProjects/FYP1/data/dataset/test/'
train_target = 'C:/Users/User/PycharmProjects/FYP1/data/processedDataset/train'
test_target = 'C:/Users/User/PycharmProjects/FYP1/data/processedDataset/test/'

# for i in range(30):
#     os.mkdir(os.path.join(train_target, str(i)))
#     os.mkdir(os.path.join(test_target, str(i)))

train_count = 0
for root, dirs, files in os.walk(train_root):
    for dir_name in dirs:
        for dir_root, subdirs, dir_files in os.walk(os.path.join(root, dir_name)):
            for file_name in dir_files:
                img = Image.open(os.path.join(dir_root, file_name))
                result = process(img)
                file_path = os.path.join(dir_root, file_name)
                new_name = dir_name + 'u' + file_name
                location = os.path.join(train_target, str(dir_name), new_name)
                shutil.copy(file_path, location)
                result.save(location)
                train_count += 1
print(str(train_count) + " training examples")


test_count = 0
for root, dirs, files in os.walk(test_root):
    for dir_name in dirs:
        for dir_root, subdirs, dir_files in os.walk(os.path.join(root, dir_name)):
            for file_name in dir_files:
                img = Image.open(os.path.join(dir_root, file_name))
                result = process(img)
                file_path = os.path.join(dir_root, file_name)
                new_name = dir_name + file_name
                location = os.path.join(test_target, str(dir_name), new_name)
                shutil.copy(file_path, location)
                result.save(location)
                test_count += 1
print(str(test_count) + " testing examples")