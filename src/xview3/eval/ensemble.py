import json
import math
import numpy as np
import os.path
import pandas as pd
import skimage.io
import sys

from xview3.utils.grid_index import GridIndex

distance_thresh = 10

def merge(preds):
    for i, pred in enumerate(preds):
        if 'input_idx' in pred.columns:
            pred = pred.drop(columns=['input_idx'])
        pred.insert(len(pred.columns), 'input_idx', [i]*len(pred))

    pred = pd.concat(preds).reset_index()

    new_scores = {}

    for scene_id in pred.scene_id.unique():
        print(scene_id)
        cur = pred[pred.scene_id == scene_id]

        # Insert into grid index.
        grid_index = GridIndex(distance_thresh)
        for index, row in cur.iterrows():
            grid_index.insert((row.detect_scene_row, row.detect_scene_column), index)

        # Set score of each point to the average over highest-scoring neighbors from each input_idx.
        for index, row in cur.iterrows():
            rect = [
                row.detect_scene_row - distance_thresh,
                row.detect_scene_column - distance_thresh,
                row.detect_scene_row + distance_thresh,
                row.detect_scene_column + distance_thresh,
            ]
            best = [0.0]*len(preds)

            for other_index in grid_index.search(rect):
                other = pred.loc[other_index]

                dx = other.detect_scene_column - row.detect_scene_column
                dy = other.detect_scene_row - row.detect_scene_row
                distance = math.sqrt(dx*dx+dy*dy)
                if distance > distance_thresh:
                    continue

                best[other.input_idx] = max(best[other.input_idx], other.score)

            best[row.input_idx] = row.score
            new_scores[index] = np.mean(best)

    print('set scores')
    for index, score in new_scores.items():
        pred.loc[index, 'score'] = score

    pred = pred.drop(columns=['index'])
    return pred

if __name__ == "__main__":
    in_paths = sys.argv[1:-1]
    out_path = sys.argv[-1]

    preds = []
    for i, in_path in enumerate(in_paths):
        pred = pd.read_csv(in_path)
        preds.append(pred)

    pred = merge(preds)
    print('save')
    pred.to_csv(out_path, index=False)
