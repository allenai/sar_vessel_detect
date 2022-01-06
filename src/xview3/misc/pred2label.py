# Add missing columns to convert prediction CSV to a label CSV.

import json
import os.path
import pandas as pd
import sys

in_path = sys.argv[1]
chips_path = sys.argv[2]
out_path = sys.argv[3]

chip_size = 800

pred = pd.read_csv(in_path)

pred.insert(len(pred.columns), 'confidence', ['HIGH']*len(pred))
pred.insert(len(pred.columns), 'source', ['manual']*len(pred))
pred.insert(len(pred.columns), 'vessel_class', [0]*len(pred))
pred.insert(len(pred.columns), 'rows', [0]*len(pred))
pred.insert(len(pred.columns), 'columns', [0]*len(pred))
pred.insert(len(pred.columns), 'chip_index', [0]*len(pred))

chip_offsets = {}
def get_offsets(scene_id):
    if scene_id not in chip_offsets:
        with open(os.path.join(chips_path, scene_id, 'coords.json'), 'r') as f:
            chip_offsets[scene_id] = json.load(f)['offsets']

    return chip_offsets[scene_id]

for index, label in pred.iterrows():
    if index%1000 == 0:
        print(index, len(pred))

    if label.is_fishing == True:
        pred.loc[index, 'vessel_class'] = 1
    elif label.is_vessel == True:
        pred.loc[index, 'vessel_class'] = 2
    else:
        pred.loc[index, 'vessel_class'] = 3

    scene_id = label.scene_id
    scene_row, scene_col = int(label.detect_scene_row), int(label.detect_scene_column)

    offsets = get_offsets(scene_id)

    # Determine which chip in the scene this point falls in.
    chip_idx = None
    for i, (start_col, start_row) in enumerate(offsets):
        if scene_row >= start_row and scene_row < start_row+chip_size and scene_col >= start_col and scene_col < start_col+chip_size:
            chip_idx = i
            break
    if chip_idx is None:
        raise Exception('failed to find chip for {}'.format(label))

    pred.loc[index, 'rows'] = scene_row - start_row
    pred.loc[index, 'columns'] = scene_col - start_col
    pred.loc[index, 'chip_index'] = chip_idx

pred.to_csv(out_path, index=False)
