# Change confidence=LOW for any label that isn't close to a point in the same scene in another CSV.
# This is to remove points where model has very low confidence.
# Workflow: first run pseudo-labeling steps in pred2label and such, then run this to post-process it.

import math
import pandas as pd
import sys
from tqdm import tqdm
from xview3.utils.grid_index import GridIndex

in_csv = sys.argv[1]
compare_csv = sys.argv[2]
out_csv = sys.argv[3]

distance_tol = 20
conf_threshold = 0.1

in_df = pd.read_csv(in_csv)
compare_df = pd.read_csv(compare_csv)
compare_df = compare_df[compare_df.score >= conf_threshold]

for scene_id in tqdm(in_df.scene_id.unique()):
    print(scene_id)
    grid_index = GridIndex(distance_tol)
    for _, label in compare_df[compare_df.scene_id == scene_id].iterrows():
        p = (int(label.detect_scene_row), int(label.detect_scene_column))
        grid_index.insert(p, p)
    for index, label in in_df[in_df.scene_id == scene_id].iterrows():
        p = (int(label.detect_scene_row), int(label.detect_scene_column))
        rect = [
            p[0]-distance_tol,
            p[1]-distance_tol,
            p[0]+distance_tol,
            p[1]+distance_tol,
        ]
        found = False
        for other_p in grid_index.search(rect):
            dx = p[0]-other_p[0]
            dy = p[1]-other_p[1]
            distance = math.sqrt(dx*dx+dy*dy)
            if distance < distance_tol:
                found = True
                break
        if not found:
            in_df.loc[index, 'confidence'] = 'LOW'
in_df.to_csv(out_csv, index=False)
