import pandas as pd
import sys

gt_path = sys.argv[1]
pred_path = sys.argv[2]
out_path = sys.argv[3]

gt = pd.read_csv(gt_path)
pred = pd.read_csv(pred_path)

scores = []
for _, label in gt.iterrows():
    if label.confidence in ['HIGH', 'MEDIUM']:
        scores.append(1.0)
    else:
        scores.append(0.1)
gt.insert(len(gt.columns), 'score', scores)
gt = gt.drop(columns=['detect_lat', 'detect_lon', 'distance_from_shore_km', 'top', 'left', 'bottom', 'right', 'detect_id', 'scene_rows', 'scene_cols'])

out = pd.concat([gt, pred], sort=True).reset_index()

out.insert(len(out.columns), 'detect_lat', [None]*len(out))
out.insert(len(out.columns), 'detect_lon', [None]*len(out))
out.insert(len(out.columns), 'distance_from_shore_km', [None]*len(out))
out.insert(len(out.columns), 'top', [None]*len(out))
out.insert(len(out.columns), 'left', [None]*len(out))
out.insert(len(out.columns), 'bottom', [None]*len(out))
out.insert(len(out.columns), 'right', [None]*len(out))
out.insert(len(out.columns), 'detect_id', [None]*len(out))
out.insert(len(out.columns), 'scene_rows', [None]*len(out))
out.insert(len(out.columns), 'scene_cols', [None]*len(out))

cols = [
    "detect_lat",
    "detect_lon",
    "vessel_length_m",
    "source",
    "detect_scene_row",
    "detect_scene_column",
    "is_vessel",
    "is_fishing",
    "distance_from_shore_km",
    "scene_id",
    "confidence",
    "top",
    "left",
    "bottom",
    "right",
    "detect_id",
    "vessel_class",
    "scene_rows",
    "scene_cols",
    "rows",
    "columns",
    "chip_index",
    "score",
]
out = out.reindex(cols, axis=1)

out.to_csv(out_path)
