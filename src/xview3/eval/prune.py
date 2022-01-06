import json
import math
import numpy as np
import os.path
import pandas as pd
import skimage.io

from xview3.utils.grid_index import GridIndex

def nms(pred, distance_thresh=10):
    '''
    Prune detections that are redundant due to a nearby higher-scoring detection.

    Args:
        pred (pd.DataFrame): dataframe containing detections, from inference.py
        distance_threshold (int): if points are within this threshold, only keep higher score point
    '''
    # Create table index so we can refer to rows by unique index.
    pred.reset_index()

    elim_inds = set()

    for scene_id in pred.scene_id.unique():
        cur = pred[pred.scene_id == scene_id]
        # Create grid index.
        grid_index = GridIndex(max(64, distance_thresh))
        for index, row in cur.iterrows():
            grid_index.insert((row['detect_scene_row'], row['detect_scene_column']), index)

        # Remove points that are close to a higher-confidence, not already-eliminated point.
        for index, row in cur.iterrows():
            if row.score == 1:
                continue
            rect = [
                row['detect_scene_row']-distance_thresh,
                row['detect_scene_column']-distance_thresh,
                row['detect_scene_row']+distance_thresh,
                row['detect_scene_column']+distance_thresh,
            ]
            for other_index in grid_index.search(rect):
                other = pred.loc[other_index]
                if other.score < row.score or (other.score == row.score and other_index <= index):
                    continue
                if other_index in elim_inds:
                    continue

                dx = other.detect_scene_column - row.detect_scene_column
                dy = other.detect_scene_row - row.detect_scene_row
                distance = math.sqrt(dx*dx+dy*dy)
                if distance > distance_thresh:
                    continue

                elim_inds.add(index)
                break

    print('nms: drop {} of {}'.format(len(elim_inds), len(pred)))

    return pred.drop(list(elim_inds))

def clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def bathymetry_pruning(pred, image_path, threshold=0, padding=0, opposite=False, use_max=False):
    # Create table index so we can refer to rows by unique index.
    pred.reset_index()

    bathymetries = {} # scene_id to bathymetry
    elim_inds = []
    for index, row in pred.iterrows():
        scene_id = row['scene_id']
        if scene_id not in bathymetries:
            bathymetries[scene_id] = skimage.io.imread(os.path.join(image_path, scene_id, 'bathymetry.tif'))

        r = clip(row.detect_scene_row//50, padding, bathymetries[scene_id].shape[0]-padding-1)
        c = clip(row.detect_scene_column//50, padding, bathymetries[scene_id].shape[1]-padding-1)

        if use_max:
            bathymetry = bathymetries[scene_id][r-padding:r+padding+1, c-padding:c+padding+1].max()
        else:
            bathymetry = bathymetries[scene_id][r-padding:r+padding+1, c-padding:c+padding+1].min()

        if bathymetry >= threshold:
            elim_inds.append(index)

    print('bathymetry: drop {} of {}'.format(len(elim_inds), len(pred)))

    if opposite:
        return pred.loc[elim_inds]

    return pred.drop(elim_inds)

def owimask_pruning(pred, image_path, padding=0):
    # Create table index so we can refer to rows by unique index.
    pred.reset_index()

    masks = {} # scene_id to owimask
    elim_inds = []
    for index, row in pred.iterrows():
        scene_id = row['scene_id']
        if scene_id not in masks:
            masks[scene_id] = skimage.io.imread(os.path.join(image_path, scene_id, 'owiMask.tif'))

        r = clip(row.detect_scene_row//50, padding, masks[scene_id].shape[0]-padding-1)
        c = clip(row.detect_scene_column//50, padding, masks[scene_id].shape[1]-padding-1)
        value = masks[scene_id][r-padding:r+padding+1, c-padding:c+padding+1].min()

        if value != 0:
            elim_inds.append(index)

    print('owimask: drop {} of {}'.format(len(elim_inds), len(pred)))

    return pred.drop(elim_inds)

def confidence_pruning(pred, threshold=0):
    out = pred[pred.score >= threshold]
    print('conf: {} -> {}'.format(len(pred), len(out)))
    return out

# Prune detections where chip for a certain channel doesn't appear on disk.
def prune_invalid(pred, chips_path, chip_size=800, channel='vh'):
    pred.reset_index()
    elim_inds = []
    chip_offsets = {}

    for index, row in pred.iterrows():
        scene_id = row.scene_id
        if scene_id not in chip_offsets:
            with open(os.path.join(chips_path, scene_id, 'coords.json'), 'r') as f:
                chip_offsets[scene_id] = json.load(f)['offsets']

        valid = False
        for chip_index, (chip_col, chip_row) in enumerate(chip_offsets[scene_id]):
            if row.detect_scene_column < chip_col:
                continue
            if row.detect_scene_column >= chip_col+chip_size:
                continue
            if row.detect_scene_row < chip_row:
                continue
            if row.detect_scene_row >= chip_row+chip_size:
                continue
            cur_path = os.path.join(chips_path, scene_id, channel, '{}_{}.npy'.format(chip_index, channel))
            if not os.path.exists(cur_path):
                continue
            im = np.load(cur_path)
            offset_col = row.detect_scene_column - chip_col
            offset_row = row.detect_scene_row - chip_row
            if im[offset_row, offset_col] < -30000:
                continue
            valid = True
            break

        if not valid:
            elim_inds.append(index)

    print('valid: drop {} of {}'.format(len(elim_inds), len(pred)))
    return pred.drop(elim_inds)

# Prune unless Google channel indicates water.
def prune_google(pred, chips_path, chip_size=800):
    pred.reset_index()
    elim_inds = []
    chip_offsets = {}

    for index, row in pred.iterrows():
        scene_id = row.scene_id
        if scene_id not in chip_offsets:
            with open(os.path.join(chips_path, scene_id, 'coords.json'), 'r') as f:
                chip_offsets[scene_id] = json.load(f)['offsets']

        for chip_index, (chip_col, chip_row) in enumerate(chip_offsets[scene_id]):
            if row.detect_scene_column < chip_col:
                continue
            if row.detect_scene_column >= chip_col+chip_size:
                continue
            if row.detect_scene_row < chip_row:
                continue
            if row.detect_scene_row >= chip_row+chip_size:
                continue
            cur_path = os.path.join(chips_path, scene_id, 'google', '{}_google.npy'.format(chip_index))
            if not os.path.exists(cur_path):
                continue
            im = np.load(cur_path)
            offset_col = row.detect_scene_column - chip_col
            offset_row = row.detect_scene_row - chip_row
            if np.abs(im[offset_row, offset_col] - 146.0/255) > 0.02:
                elim_inds.append(index)
            break

    print('google: drop {} of {}'.format(len(elim_inds), len(pred)))
    return pred.drop(elim_inds)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prune detections."
    )
    parser.add_argument("--nms_thresh", type=int, help="Run NMS, with this threshold", default=None)
    parser.add_argument("--bathymetry_thresh", type=int, help="Run bathymetry pruning, with this threshold", default=None)
    parser.add_argument("--bathymetry_padding", type=int, help="Check this many nearby pixels at 500m/pixel resolution during bathymetry pruning", default=0)
    parser.add_argument("--bathymetry_opposite", type=bool, help="Keep >= threshold instead of < threshold", default=False)
    parser.add_argument("--bathymetry_max", type=bool, help="Take max instead of min over bathymetry padding", default=False)
    parser.add_argument("--conf", type=float, help="Run confidence pruning, with this threshold", default=None)
    parser.add_argument("--image_folder", help="Path to the xView3 images (required for bathymetry pruning)", default=None)
    parser.add_argument("--chips_path", help="Path to chips (required for valid pruning)", default=None)
    parser.add_argument("--in_path", help="Input file")
    parser.add_argument("--out_path", help="Output file")
    parser.add_argument("--drop_cols", help="Drop columns like score and bathymetry", default=False)
    parser.add_argument("--owimask", type=bool, default=False)
    parser.add_argument("--valid", type=bool, default=False)
    parser.add_argument("--google", type=bool, default=False)

    args = parser.parse_args()

    pred = pd.read_csv(args.in_path)

    if args.nms_thresh is not None:
        pred = nms(pred, distance_thresh=args.nms_thresh)
    if args.bathymetry_thresh is not None:
        pred = bathymetry_pruning(
            pred, args.image_folder,
            threshold=args.bathymetry_thresh,
            padding=args.bathymetry_padding,
            opposite=args.bathymetry_opposite,
            use_max=args.bathymetry_max,
        )
    if args.conf is not None:
        pred = confidence_pruning(pred, threshold=args.conf)
    if args.owimask:
        pred = owimask_pruning(pred, args.image_folder, padding=args.bathymetry_padding)
    if args.valid:
        pred = prune_invalid(pred, args.chips_path)
    if args.google:
        pred = prune_google(pred, args.chips_path)

    if args.drop_cols:
        good_columns = [
            'detect_scene_row',
            'detect_scene_column',
            'scene_id',
            'is_vessel',
            'is_fishing',
            'vessel_length_m',
        ]
        bad_columns = []
        for column_name in pred.columns:
            if column_name in good_columns:
                continue
            bad_columns.append(column_name)
        pred = pred.drop(columns=bad_columns)

    pred.to_csv(args.out_path, index=False)
