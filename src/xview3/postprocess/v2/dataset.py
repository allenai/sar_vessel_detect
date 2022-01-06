import os.path
import pandas as pd
import random
import skimage.io
import torch

class Dataset(object):
    def __init__(
        self,
        csv_path='/xview3/postprocess/v2/train2/labels.csv',
        image_path='/xview3/postprocess/v2/train2/boxes/',
        filter_func=None,
    ):

        self.csv_path = csv_path
        self.image_path = image_path
        self.labels = pd.read_csv(csv_path)
        self.indices = list(range(len(self.labels)))

        if filter_func is not None:
            self.indices = [idx for idx in self.indices if filter_func(idx)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        img = skimage.io.imread(os.path.join(self.image_path, '{}.png'.format(idx)))
        img = torch.as_tensor(img).permute(2, 0, 1)
        img = img.float()/255

        # crop
        left = random.randint(0, 16)
        right = img.shape[2] - (16 - left)
        top = random.randint(0, 16)
        bottom = img.shape[1] - (16 - top)
        #left, right, top, bottom = 8, img.shape[2]-8, 8, img.shape[1]-8
        img = img[:, top:bottom, left:right]
        img = torch.nn.functional.pad(img, (8, 8, 8, 8))

        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])

        label = self.labels.loc[idx]
        target = {}

        if label.vessel_length_m > 0:
            target['vessel_length'] = label.vessel_length_m
        else:
            target['vessel_length'] = -1.0

        if label.confidence == 'HIGH':
            target['confidence'] = 2
        elif label.confidence == 'MEDIUM':
            target['confidence'] = 1
        elif label.confidence == 'LOW':
            target['confidence'] = 0
        else:
            target['confidence'] = -1

        if label.correct == True and label.confidence in ['HIGH', 'MEDIUM']:
            target['correct'] = 1
        else:
            target['correct'] = 0

        if label.source == 'ais':
            target['source'] = 0
        elif label.source == 'manual':
            target['source'] = 1
        elif label.source == 'ais/manual':
            target['source'] = 2
        else:
            target['source'] = -1

        if label.is_vessel == True and label.is_fishing == True:
            target['fishing'] = 1
        elif label.is_vessel == True and label.is_fishing == False:
            target['fishing'] = 0
        else:
            target['fishing'] = -1

        if label.is_vessel == True:
            target['vessel'] = 1
        elif label.is_vessel == False:
            target['vessel'] = 0
        else:
            target['vessel'] = -1

        return img, target, idx
