import pandas as pd
import sys

from xview3.utils.grid_index import GridIndex
from xview3.utils import coords

gt_path = sys.argv[1]
pred_path = sys.argv[2]
scene_path = sys.argv[3]
out_path = sys.argv[4]

distance_tol = 20

gt = pd.read_csv(gt_path)
pred = pd.read_csv(pred_path)

with open(scene_path, 'r') as f:
    scene_ids = [line.strip() for line in f.readlines() if line.strip()]

print('prepare gt')
gt.insert(len(gt.columns), 'correct', [True]*len(gt))
gt = gt[['scene_id', 'detect_scene_row', 'detect_scene_column', 'vessel_length_m', 'confidence', 'correct', 'source', 'is_fishing', 'is_vessel']]
gt = gt[gt.scene_id.isin(scene_ids)]

print('prepare pred')
pred = pred[['scene_id', 'detect_scene_row', 'detect_scene_column']]
pred.insert(len(pred.columns), 'vessel_length_m', [None]*len(pred))
pred.insert(len(pred.columns), 'confidence', [None]*len(pred))
pred.insert(len(pred.columns), 'correct', [False]*len(pred))
pred.insert(len(pred.columns), 'source', [None]*len(pred))
pred.insert(len(pred.columns), 'is_fishing', [None]*len(pred))
pred.insert(len(pred.columns), 'is_vessel', [None]*len(pred))
pred = pred[['scene_id', 'detect_scene_row', 'detect_scene_column', 'vessel_length_m', 'confidence', 'correct', 'source', 'is_fishing', 'is_vessel']]
pred = pred[pred.scene_id.isin(scene_ids)]

def get_point(label):
    return coords.Point(int(label.detect_scene_column), int(label.detect_scene_row))

print('index gt')
grid_indexes = {}
for index, label in gt.iterrows():
    scene_id = label.scene_id
    if scene_id not in grid_indexes:
        grid_indexes[scene_id] = GridIndex(32)
    p = get_point(label)
    grid_indexes[scene_id].insert((p.x, p.y), index)

# Eliminate predictions that are correct.
print('prune correct pred')
def is_incorrect(label):
    p = get_point(label)
    rect = [
        p.x - distance_tol,
        p.y - distance_tol,
        p.x + distance_tol,
        p.y + distance_tol,
    ]
    for other_index in grid_indexes[label.scene_id].search(rect):
        other_label = gt.loc[other_index]
        other_p = get_point(other_label)
        if p.distance(other_p) < distance_tol:
            return False
    return True

elim_inds = []
for index, label in pred.iterrows():
    if is_incorrect(label):
        continue
    elim_inds.append(index)
pred = pred.drop(elim_inds)

print('concat')
df = pd.concat([gt, pred])
df.to_csv(out_path, index=False)
