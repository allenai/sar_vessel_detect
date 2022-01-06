import json
import numpy
import os.path
import pandas as pd
import sys
from tqdm import tqdm
import torch

from xview3.postprocess.v2.model_simple import Model
from xview3.transforms import CustomNormalize3

model_path = sys.argv[1]
csv_path = sys.argv[2]
chips_path = sys.argv[3]
out_path = sys.argv[4]
mode = sys.argv[5]

if mode not in ['length', 'full', 'attribute']:
    raise Exception('unknown mode {}'.format(mode))

batch_size = 32
num_loader_workers = 4
chip_size = 800
crop_size = 128

class Dataset(object):
    def __init__(self, csv_path, chips_path):
        self.csv_path = csv_path
        self.chips_path = chips_path
        self.labels = pd.read_csv(csv_path)

        print('indexing chip by offsets')
        # Map from (scene_id, chip_row, chip_col) to chip index.
        self.chip_by_pos = {}

        for scene_id in self.labels.scene_id.unique():
            with open(os.path.join(self.chips_path, scene_id, 'coords.json'), 'r') as f:
                cur_offsets = json.load(f)['offsets']

            # Add this chip to chip_by_pos.
            for chip_idx, (chip_col, chip_row) in enumerate(cur_offsets):
                self.chip_by_pos[(scene_id, chip_row//chip_size, chip_col//chip_size)] = chip_idx

        print('bucketing labels by chip')
        # Map from (scene_id, chip_row, chip_col) to pred indices that fall in that chip.
        self.chip_to_indices = {}

        for index, label in self.labels.iterrows():
            scene_id = label.scene_id
            row, col = int(label.detect_scene_row), int(label.detect_scene_column)
            chip_key = (scene_id, row//chip_size, col//chip_size)
            if chip_key not in self.chip_to_indices:
                self.chip_to_indices[chip_key] = []
            self.chip_to_indices[chip_key].append(index)

        self.chips = list(self.chip_to_indices.keys())
        print('need {} chips'.format(len(self.chips)))
        self.transform = CustomNormalize3({'channels': ['vh', 'vv', 'bathymetry']})

    def __len__(self):
        return len(self.chips)

    def load_or_zeros(self, path):
        if os.path.exists(path):
            return numpy.load(path)
        else:
            return -32768*numpy.ones((chip_size, chip_size), dtype=numpy.float32)

    def __getitem__(self, i):
        scene_id, chip_row, chip_col = self.chips[i]
        indices = self.chip_to_indices[(scene_id, chip_row, chip_col)]

        centers = []
        for label_idx in indices:
            label = self.labels.loc[label_idx]
            centers.append((
                label.detect_scene_row - chip_row*chip_size,
                label.detect_scene_column - chip_col*chip_size,
            ))

        scene_dir = os.path.join(self.chips_path, scene_id)
        img = torch.zeros((3, 3*chip_size, 3*chip_size), dtype=torch.float32)

        for row_offset in [-1, 0, 1]:
            for col_offset in [-1, 0, 1]:
                x1 = col_offset*chip_size - crop_size//2
                y1 = row_offset*chip_size - crop_size//2
                x2 = (col_offset+1)*chip_size + crop_size//2
                y2 = (row_offset+1)*chip_size + crop_size//2
                needed = any([
                    row >= y1 and
                    row < y2 and
                    col >= x1 and
                    col < x2
                    for row, col in centers
                ])
                if not needed:
                    continue

                chip_idx = self.chip_by_pos.get((scene_id, chip_row + row_offset, chip_col + col_offset))
                if chip_idx is None:
                    continue

                vh_im = self.load_or_zeros(os.path.join(scene_dir, 'vh/{}_vh.npy'.format(chip_idx)))
                vv_im = self.load_or_zeros(os.path.join(scene_dir, 'vv/{}_vv.npy'.format(chip_idx)))
                bathymetry = self.load_or_zeros(os.path.join(scene_dir, 'bathymetry/{}_bathymetry.npy'.format(chip_idx)))
                cur_img = numpy.stack([vh_im, vv_im, bathymetry], axis=0)
                cur_img = torch.as_tensor(cur_img)
                img[:, (1+row_offset)*chip_size:(2+row_offset)*chip_size, (1+col_offset)*chip_size:(2+col_offset)*chip_size] = cur_img

        img, _ = self.transform(img, None)

        crops = []
        for row, col in centers:
            crop = img[:, row+chip_size-crop_size//2:row+chip_size+crop_size//2, col+chip_size-crop_size//2:col+chip_size+crop_size//2]
            crop = crop[:, 8:120, 8:120]
            crop = torch.nn.functional.pad(crop, (8, 8, 8, 8))
            crops.append(crop)

        return torch.stack(crops, dim=0), torch.tensor(indices, dtype=torch.int)

dataset = Dataset(csv_path=csv_path, chips_path=chips_path)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=None,
    num_workers=num_loader_workers,
)

model = Model()
model.load_state_dict(torch.load(model_path))
device = torch.device("cuda")
model.to(device)

df = pd.read_csv(csv_path)
elim_inds = []

with torch.no_grad():
    model.eval()
    for images, indexes in tqdm(data_loader):
        images = images.to(device)
        t = model(images)
        t = [x.cpu() for x in t]
        pred_length, pred_confidence, pred_correct, pred_source, pred_fishing, pred_vessel = t

        for i, index in enumerate(indexes):
            index = index.item()

            # Prune confidence=LOW.
            if pred_confidence[i, :].argmax() == 0 and mode == 'full':
                elim_inds.append(index)
                continue

            df.loc[index, 'vessel_length_m'] = pred_length[i].item()

            if mode in ['full', 'attribute']:
                df.loc[index, 'fishing_score'] = pred_fishing[i, 1].item()
                df.loc[index, 'vessel_score'] = pred_vessel[i, 1].item()
                df.loc[index, 'low_score'] = pred_confidence[i, 0].item()
                df.loc[index, 'is_fishing'] = (pred_fishing[i, 1] > 0.5).item() & (pred_vessel[i, 1] > 0.5).item()
                df.loc[index, 'is_vessel'] = (pred_vessel[i, 1] > 0.5).item()
                df.loc[index, 'correct_score'] = pred_correct[i, 1].item()

            if mode == 'full':
                df.loc[index, 'score'] = pred_correct[i, 1].item()

df = df.drop(elim_inds)
df.to_csv(out_path, index=False)
