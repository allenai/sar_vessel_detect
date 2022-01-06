# Like visualize_label_boxes but extracts images from pre-processed chips so it's much faster.

import csv
import json
import multiprocessing
import numpy
import os, os.path
import skimage.io, skimage.transform
import sys
import torch

from xview3.transforms import CustomNormalize3

csv_path = sys.argv[1]
chip_path = sys.argv[2]
out_path = sys.argv[3]

chip_size = 800
crop_size = 128

# Load labels.
labels = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append(row)

# Add indices.
for i, label in enumerate(labels):
    label['index'] = i

# Annotate each label with its chip index.
chip_offsets = {}
def get_offsets(scene_id):
    if scene_id not in chip_offsets:
        with open(os.path.join(chip_path, scene_id, 'coords.json'), 'r') as f:
            chip_offsets[scene_id] = json.load(f)['offsets']
    return chip_offsets[scene_id]

for label in labels:
    scene_id = label['scene_id']
    row, col = int(label['detect_scene_row']), int(label['detect_scene_column'])
    offsets = get_offsets(scene_id)

    # Determine which chip in the scene this point falls in.
    chip_idx = None
    for i, (start_col, start_row) in enumerate(offsets):
        if row >= start_row and row < start_row+chip_size and col >= start_col and col < start_col+chip_size:
            chip_idx = i
            break
    if chip_idx is None:
        raise Exception('failed to find chip for {}'.format(label))
    label['chip_index'] = chip_idx
    label['chip_row'] = start_row
    label['chip_col'] = start_col
    label['detect_chip_row'] = row - start_row
    label['detect_chip_col'] = col - start_col

# Sort by chips, load all chips we need into memory.
labels_by_chip = {}
for label in labels:
    k = (label['scene_id'], label['chip_index'])
    if k not in labels_by_chip:
        labels_by_chip[k] = []
    labels_by_chip[k].append(label)

transform = CustomNormalize3({'channels': ['vh', 'vv', 'bathymetry']})

def f(t):
    scene_id, chip_index, cur_labels = t

    dir = os.path.join(chip_path, scene_id)
    vh_im = numpy.load(os.path.join(dir, 'vh/{}_vh.npy'.format(chip_index)))
    vv_im = numpy.load(os.path.join(dir, 'vv/{}_vv.npy'.format(chip_index)))
    bathymetry = numpy.load(os.path.join(dir, 'bathymetry/{}_bathymetry.npy'.format(chip_index)))
    img = numpy.stack([vh_im, vv_im, bathymetry], axis=0)
    img = torch.tensor(img, dtype=torch.float32)
    img, _ = transform(img, None)
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img = numpy.clip(img*255, 0, 255).astype('uint8')
    img = numpy.pad(img, pad_width=[(crop_size//2, crop_size//2), (crop_size//2, crop_size//2), (0, 0)])

    for label in cur_labels:
        out_fname = os.path.join(out_path, '{}.png'.format(label['index']))
        if os.path.exists(out_fname):
            continue

        crop_row = label['detect_chip_row']
        crop_col = label['detect_chip_col']
        crop = img[crop_row:crop_row+crop_size, crop_col:crop_col+crop_size, :]
        skimage.io.imsave(out_fname, crop)

p = multiprocessing.Pool(16)
inputs = [(scene_id, chip_index, cur_labels) for (scene_id, chip_index), cur_labels in labels_by_chip.items()]
p.map(f, inputs)
p.close()
