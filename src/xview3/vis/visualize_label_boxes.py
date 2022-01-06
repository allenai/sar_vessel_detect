# Script that randomly samples annotations and visualizes 64x64 crops around them.

import csv
import numpy
import os, os.path
import random
import skimage.io, skimage.transform
import sys

csv_path = sys.argv[1]
scene_path = sys.argv[2]
count = int(sys.argv[3])
out_path = sys.argv[4]

# Customize this function to sample different types of labels.
def udf(label):
    return label['is_vessel'] == 'True' and label['is_fishing'] == 'True'
    #return label['is_vessel'] == 'True' and label['is_fishing'] == 'False'
    #return label['is_vessel'] == 'False'

# Check what scenes are available.
scene_ids = os.listdir(scene_path)
scene_ids = [fname for fname in scene_ids if '.tar.gz' not in fname]
scene_ids = set(scene_ids)

# Load labels. Only use labels in the available scenes.
labels = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append(row)
labels = [label for label in labels if label['scene_id'] in scene_ids]

# Sample labels that satisfy the user-defined function.
labels = [label for label in labels if udf(label)]
labels = random.sample(labels, count)

# Load all scenes we need into memory.
needed_scenes = set([label['scene_id'] for label in labels])
scene_images = {}
for scene_id in needed_scenes:
    print('loading', scene_id)
    dir = os.path.join(scene_path, scene_id)
    vv_im = skimage.io.imread(os.path.join(dir, 'VV_dB.tif'))
    vh_im = skimage.io.imread(os.path.join(dir, 'VH_dB.tif'))
    bathymetry_im = skimage.io.imread(os.path.join(dir, 'bathymetry.tif'))
    bathymetry_im = skimage.transform.resize(bathymetry_im, vv_im.shape, preserve_range=True).astype('float16')
    scene_im = numpy.stack([vv_im, vh_im, bathymetry_im], axis=2)
    #scene_im = numpy.clip((scene_im+32770)/135, 0, 255).astype('uint8')
    scene_images[scene_id] = scene_im

def normalize_channel(im):
    im = im-im.min()
    return numpy.clip(im/im.max()*255, 0, 255).astype('uint8')

def normalize_image(im):
    return numpy.stack([
        normalize_channel(im[:, :, 0]),
        normalize_channel(im[:, :, 1]),
        normalize_channel(im[:, :, 0]),
        #normalize_channel(im[:, :, 2]),
    ], axis=2)

# Extract crops and output.
for i, label in enumerate(labels):
    scene_im = scene_images[label['scene_id']]
    row, col = int(label['detect_scene_row']), int(label['detect_scene_column'])
    crop = scene_im[row-32:row+32, col-32:col+32, :]
    crop = numpy.stack([
        crop[:, :, 0],
        crop[:, :, 0],
        crop[:, :, 1],
    ], axis=2)
    crop = numpy.clip((crop+60)*(255/65), 0, 255).astype('uint8')
    skimage.io.imsave(os.path.join(out_path, '{}.jpg'.format(i)), crop)

# Code to create combined images.
'''
import numpy
import os
import skimage.io
for d in ['fishing', 'nonfishing', 'nonvessel']:
    im = numpy.zeros((512, 512, 3), dtype='uint8')
    for fname in os.listdir(d):
        id = int(fname.split('.jpg')[0])
        x = id//8
        y = id%8
        im[64*x:64*(x+1), 64*y:64*(y+1), :] = skimage.io.imread(d+'/'+fname)
    skimage.io.imsave(d+'/tile.jpg', im)
'''
