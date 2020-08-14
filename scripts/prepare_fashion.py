"""
Run this script to prepare the miniImageNet dataset.

This script uses the 100 classes of 600 images each used in the Matching
Networks paper. The exact images used are given in data/mini_imagenet.txt which
is downloaded from the link provided in the paper (https://goo.gl/e3orz6).

1. Download files from:
    https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view
    and place in data/miniImageNet/images
2. Run the script
"""
from tqdm import tqdm as tqdm
import shutil
import os
import sys
import argparse
sys.path.append('./')
import pdb

from config import DATA_PATH
from few_shot.utils import mkdir, rmdir

parser = argparse.ArgumentParser()
parser.add_argument('--size', default='small', help='Size of input image')
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
    if os.path.exists(src):
        dst = DATA_PATH + 'fashion-dataset/{}/{}/{}'.format(subset_folder, image_category, image_id + '.jpg')
        shutil.copy(src, dst)

print('Processing fashion_{} finished'.format(args.size))
