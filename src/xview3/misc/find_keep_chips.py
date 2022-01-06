# Make list of chips that either have a ground truth label,
# or that have a prediction at a low confidence threshold.
# Basically, we don't want to waste time (and parameters) training the model on background chips that definitely don't have anything.

import json
import pandas as pd
import sys

gt_path = sys.argv[1]
pred_path = sys.argv[2]
train_scene_path = sys.argv[3]
out_path = sys.argv[4]

gt = pd.read_csv(gt_path)
pred = pd.read_csv(pred_path)
with open(train_scene_path, 'r') as f:
    train_scenes = [line.strip() for line in f.readlines() if line.strip()]

keep_chips = set()
for _, label in gt[gt.scene_id.isin(train_scenes)].iterrows():
    keep_chips.add((label.scene_id, int(label.chip_index)))
for _, label in pred.iterrows():
    keep_chips.add((label.scene_id, int(label.chip_index)))

with open(out_path, 'w') as f:
    json.dump(list(keep_chips), f)
