# Visualize either ground truth labels or inferred CSVs, doesn't matter which.

import argparse
import json
import numpy
import os, os.path
import pandas as pd
import random
import skimage.io, skimage.transform
import sys


parser = argparse.ArgumentParser(
    description="Visualize points in a label or inferred CSV file, using chips."
)

parser.add_argument("--csv_path", help="Path to points")
parser.add_argument("--chip_path", help="Path to chips")
parser.add_argument("--out_path", help="Output path for PNGs")

parser.add_argument("--count", type=int, help="Number of points to randomly sample for visualization (-1 for all points)", default=None)
parser.add_argument("--chip", help="A specific chip to visualize, of the form sceneID_row_col", default=None)

args = parser.parse_args()

chip_size = 800
box_size = 15

def clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

# Load labels.
with open(args.csv_path, 'r') as f:
    labels = pd.read_csv(f)

chip_offsets = {}
def get_offsets(scene_id):
    if scene_id not in chip_offsets:
        with open(os.path.join(args.chip_path, scene_id, 'coords.json'), 'r') as f:
            chip_offsets[scene_id] = json.load(f)['offsets']
    return chip_offsets[scene_id]

# Sample a subset, and determine which chips to visualize based on that subset.
if args.count:
    if args.count == -1:
        subset = labels
    else:
        subset = labels.sample(n=args.count)

    vis_chips = set()
    for _, label in subset.iterrows():
        scene_id = label.scene_id
        row, col = int(label.detect_scene_row), int(label.detect_scene_column)

        offsets = get_offsets(scene_id)

        # Determine which chip in the scene this point falls in.
        chip_idx = None
        for i, (start_col, start_row) in enumerate(offsets):
            if row >= start_row and row < start_row+chip_size and col >= start_col and col < start_col+chip_size:
                chip_idx = i
                break
        if chip_idx is None:
            raise Exception('failed to find chip for {}'.format(p))
        vis_chips.add((scene_id, chip_idx, start_row, start_col))

    vis_chips = list(vis_chips)
    random.shuffle(vis_chips)
elif args.chip:
    scene_id, row, col = args.chip.split('_')
    row = int(row)
    col = int(col)

    chip_idx = None
    offsets = get_offsets(scene_id)
    for i, (start_col, start_row) in enumerate(offsets):
        if row >= start_row and row < start_row+chip_size and col >= start_col and col < start_col+chip_size:
            chip_idx = i
            break
    if chip_idx is None:
        raise Exception('failed to find chip for {}'.format(args.chip))
    vis_chips = [(scene_id, chip_idx, start_row, start_col)]

# Visualize the selected chips.
for i, (scene_id, chip_idx, start_row, start_col) in enumerate(vis_chips):
    # Load background.
    dir = os.path.join(args.chip_path, scene_id)
    vh_im = numpy.load(os.path.join(dir, 'vh/{}_vh.npy'.format(chip_idx)))
    vh_im = numpy.clip((vh_im+50)*(255/55), 0, 255).astype('uint8')
    chip_im = numpy.stack([vh_im]*3, axis=2)

    # Draw all points that fall in this chip.
    for _, label in labels[labels.scene_id == scene_id].iterrows():
        row = label.detect_scene_row - start_row
        col = label.detect_scene_column - start_col
        if row < 0 or row >= chip_size or col < 0 or col >= chip_size:
            continue
        row = clip(row, box_size, chip_size-box_size)
        col = clip(col, box_size, chip_size-box_size)

        if label.is_fishing is True:
            color = [255, 0, 255]
        elif label.is_vessel is True:
            color = [255, 255, 255]
        else:
            color = [0, 0, 255]

        chip_im[row-box_size:row+box_size, col-box_size:col-box_size+1] = color
        chip_im[row-box_size:row+box_size, col+box_size-1:col+box_size] = color
        chip_im[row-box_size:row-box_size+1, col-box_size:col+box_size] = color
        chip_im[row+box_size-1:row+box_size, col-box_size:col+box_size] = color

    skimage.io.imsave(os.path.join(args.out_path, '{}_{}_{}_{}.png'.format(i, scene_id, start_row, start_col)), chip_im)
