# Visualize true positives, false positives, and false negatives produced by metric.py.

import json
import numpy
import os, os.path
import random
import skimage.io, skimage.transform
import sys

json_path = sys.argv[1]
chip_path = sys.argv[2]
count = int(sys.argv[3])
out_path = sys.argv[4]

chip_size = 800
box_size = 15

def clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

# Load outputs.
with open(json_path, 'r') as f:
    meta = json.load(f)

def pick_class_color(is_vessel, is_fishing):
    if is_fishing is True:
        # magenta
        return [255, 0, 255]
    if is_vessel is True:
        # white
        return [255, 255, 255]
    # blue
    return [0, 0, 255]

def pick_confidence_color(confidence):
    if confidence == 'LOW':
        return [0, 0, 0]
    if confidence == 'MEDIUM':
        return [128, 128, 128]
    if confidence == 'HIGH':
        return [255, 255, 255]
    raise Exception('unknown confidence ' + confidence)

def pick_source_color(source):
    if source == 'ais':
        return [0, 0, 0]
    if source == 'manual':
        return [128, 128, 128]
    if source == 'ais/manual':
        return [255, 255, 255]
    raise Exception('unknown source ' + source)

def get_point(k, p):
    if k == 'tp':
        return {
            'scene_id': p[0][0],
            'row': p[0][1],
            'col': p[0][2],
            'gt_color': pick_class_color(p[0][3], p[0][4]),
            'confidence_color': pick_confidence_color(p[0][5]),
            'source_color': pick_source_color(p[0][6]),
            'pred_color': pick_class_color(p[1][3], p[1][4]),
        }
    elif k == 'fp':
        return {
            'scene_id': p[0],
            'row': p[1],
            'col': p[2],
            'gt_color': None,
            'confidence_color': None,
            'source_color': None,
            'pred_color': pick_class_color(p[3], p[4]),
        }
    elif k == 'fn':
        return {
            'scene_id': p[0],
            'row': p[1],
            'col': p[2],
            'gt_color': pick_class_color(p[3], p[4]),
            'confidence_color': pick_confidence_color(p[5]),
            'source_color': pick_source_color(p[6]),
            'pred_color': None,
        }
    return None

# Sample count of each of tp, fn, and fp.
# We will visualize the chips containing each of these.
vis_points = []
for k in ['tp', 'fp', 'fn']:
    points = random.sample(meta[k], count)
    vis_points.extend([get_point(k, p) for p in points])

# Translate vis_points to chips.
vis_chips = set()
scene_coords = {} # map from scene_id to chip coordinates in that scene
for p in vis_points:
    scene_id = p['scene_id']
    if scene_id not in scene_coords:
        with open(os.path.join(chip_path, scene_id, 'coords.json'), 'r') as f:
            scene_coords[scene_id] = json.load(f)['offsets']
    # Determine which chip in the scene this point falls in.
    chip_idx = None
    for i, (start_col, start_row) in enumerate(scene_coords[scene_id]):
        if p['row'] >= start_row and p['row'] < start_row+chip_size and p['col'] >= start_col and p['col'] < start_col+chip_size:
            chip_idx = i
            break
    if chip_idx is None:
        raise Exception('failed to find chip for {}'.format(p))
    vis_chips.add((scene_id, chip_idx))

vis_chips = list(vis_chips)
random.shuffle(vis_chips)

# Visualize the selected chips.
for i, (scene_id, chip_idx) in enumerate(vis_chips):
    # Load background.
    dir = os.path.join(chip_path, scene_id)
    #vv_im = numpy.load(os.path.join(dir, 'vv/{}_vv.npy'.format(chip_idx)))
    vh_im = numpy.load(os.path.join(dir, 'vh/{}_vh.npy'.format(chip_idx)))
    #vv_im = numpy.clip((vv_im+50)*(255/55), 0, 255).astype('uint8')
    vh_im = numpy.clip((vh_im+50)*(255/55), 0, 255).astype('uint8')
    chip_im = numpy.stack([vh_im, vh_im, vh_im], axis=2)

    # Draw all points that fall in this chip.
    start_col, start_row = scene_coords[scene_id][chip_idx]
    for k, color in [('tp', [0, 255, 0]), ('fp', [0, 255, 255]), ('fn', [255, 255, 0])]:
        for p in meta[k]:
            p = get_point(k, p)
            if p['scene_id'] != scene_id:
                continue
            row = p['row'] - start_row
            col = p['col'] - start_col
            if row < 0 or row >= chip_size or col < 0 or col >= chip_size:
                continue
            row = clip(row, box_size, chip_size-box_size)
            col = clip(col, box_size, chip_size-box_size)
            chip_im[row-box_size:row+box_size, col-box_size:col-box_size+1] = color
            chip_im[row-box_size:row+box_size, col+box_size-1:col+box_size] = color
            chip_im[row-box_size:row-box_size+1, col-box_size:col+box_size] = color
            chip_im[row+box_size-1:row+box_size, col-box_size:col+box_size] = color
            if p['gt_color']:
                chip_im[row-box_size:row-box_size+5, col-box_size:col-box_size+5, :] = p['gt_color']
            if p['pred_color']:
                chip_im[row+box_size-5:row+box_size, col+box_size-5:col+box_size, :] = p['pred_color']
            if p['confidence_color']:
                chip_im[row-box_size:row-box_size+5, col+box_size-5:col+box_size, :] = p['confidence_color']
            if p['source_color']:
                chip_im[row+box_size-5:row+box_size, col-box_size:col-box_size+5, :] = p['source_color']

    skimage.io.imsave(os.path.join(out_path, '{}_{}_{}_{}.png'.format(i, scene_id, start_row, start_col)), chip_im)
