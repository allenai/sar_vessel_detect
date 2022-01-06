from xview3.processing.dataloader import SARDataset
from xview3.transforms import CustomNormalize3, get_transforms

import numpy
import os
import random
import skimage.io
import sys

chips_path = sys.argv[1]
scene_path = sys.argv[2]
channels = sys.argv[3].strip().split(',')
out_path = sys.argv[4]

dataset = SARDataset(
    chips_path=chips_path,
    scene_path=scene_path,
    transforms=CustomNormalize3({'channels': channels}),
    channels=channels,
    i2=True,
)

indices = random.sample(list(range(len(dataset))), 64)
for i, index in enumerate(indices):
    img, _ = dataset[index]
    #for channel_idx, channel in enumerate(channels):
    for channel_idx, channel in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]:
        if channel_idx >= img.shape[0]:
            continue
        cur = numpy.clip(img[channel_idx, :, :].numpy()*255, 0, 255).astype('uint8')
        skimage.io.imsave(os.path.join(out_path, '{}_{}.png'.format(i, channel)), cur)
