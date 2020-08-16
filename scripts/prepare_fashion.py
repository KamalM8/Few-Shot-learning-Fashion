"""
Run this script to prepare the fashion dataset.

This script uses the metaTrain.txt and metaTest.txt to create a custom
training and validation class split respectively. Its large version could
be downloaded from:
(https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1)
and the small version could be downloaded from:
(https://www.kaggle.com/paramaggarwal/fashion-product-images-small)

Follow steps in README.md (Training Data section) for more information
"""

from tqdm import tqdm as tqdm
import shutil
import os
import argparse
from PIL import Image

from config import DATA_PATH
from few_shot.utils import mkdir, rmdir

import sys
sys.path.append('./')

parser = argparse.ArgumentParser()
parser.add_argument('--size', default='small', help='Dataset size: small or large (default: small)')
args = parser.parse_args()

# Clean up folders
rmdir(DATA_PATH + '/fashion-dataset/images_background')
rmdir(DATA_PATH + '/fashion-dataset/images_evaluation')
mkdir(DATA_PATH + '/fashion-dataset/images_background')
mkdir(DATA_PATH + '/fashion-dataset/images_evaluation')

# Find class identities
classes = []
meta_file = open(DATA_PATH + '/fashion-dataset/styles.csv', 'r')
meta_data = meta_file.readlines()[1:]
meta_train_file = open(DATA_PATH + '/fashion-dataset/metaTrain.txt','r')
meta_test_file = open(DATA_PATH + '/fashion-dataset/metaTest.txt','r')
meta_train_data = set([line.rstrip() for line in meta_train_file])
meta_test_data = set([line.rstrip() for line in meta_test_file])

# Train/test split
background_classes, evaluation_classes = meta_train_data, meta_test_data

# Create class folders
for c in background_classes:
    mkdir(DATA_PATH + '/fashion-dataset/images_background/{}/'.format(c))

for c in evaluation_classes:
    mkdir(DATA_PATH + '/fashion-dataset/images_evaluation/{}/'.format(c))

def resize(path):
    img = Image.open(path)
    img = img.resize((300,300))
    return img

# Move images to correct location
root = DATA_PATH + 'fashion-dataset/images_{}'.format(args.size)
for line in tqdm(meta_data):
    entry = line.split(',')
    image_id = entry[0]
    image_category = entry[4]
    # Send to correct folder
    if image_category in evaluation_classes:
        subset_folder = 'images_evaluation'
    elif image_category in background_classes:
        subset_folder = 'images_background'
    else:
        continue

    src = '{}/{}'.format(root, image_id + '.jpg')
    dst = DATA_PATH + 'fashion-dataset/{}/{}/{}'.format(subset_folder, image_category, image_id + '.jpg')
    if os.path.exists(src):
        if args.size == 'small':
            shutil.copy(src, dst)
        elif args.size == 'large':
            resized_img = resize(src)
            resized_img.save(dst)

print('Processing fashion_{} finished'.format(args.size))
