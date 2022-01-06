from xview3.processing.dataloader import SARDataset
import xview3.transforms

import numpy
import os
import random
import skimage.io
import sys

chips_path = sys.argv[1]
scene_path = sys.argv[2]
channels = sys.argv[3].strip().split(',')
transform_names = sys.argv[4].strip().split(',')
out_path = sys.argv[5]

transforms = xview3.transforms.get_transforms(transform_names, {
    'channels': channels,
    'bbox_size': 10,
})

dataset = SARDataset(
    chips_path=chips_path,
    scene_path=scene_path,
    transforms=transforms,
    channels=channels,
    clip_boxes=True,
)

indices = random.sample(list(range(len(dataset))), 64)
for i, index in enumerate(indices):
    img, targets = dataset[index]
    img = numpy.clip(img.numpy().transpose(1, 2, 0)*255, 0, 255).astype('uint8')
    for box in targets['boxes']:
        left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img[top:bottom, left:left+2, :] = [255, 0, 0]
        img[top:bottom, right-2:right, :] = [255, 0, 0]
        img[top:top+2, left:right, :] = [255, 0, 0]
        img[bottom-2:bottom, left:right, :] = [255, 0, 0]
    skimage.io.imsave(os.path.join(out_path, '{}.png'.format(i)), img)
