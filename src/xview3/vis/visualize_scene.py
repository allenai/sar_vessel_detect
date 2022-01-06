# Convert scene to png.

import json
import numpy
import os, os.path
import random
import skimage.io, skimage.transform
import sys

scene_dir = sys.argv[1]
out_fname = sys.argv[2]

im = skimage.io.imread(os.path.join(scene_dir, 'VH_dB.tif'))[8192:8192+4096, 8192:8192+4096]
im = numpy.clip((im+50)*(255/55), 0, 255).astype('uint8')
skimage.io.imsave(out_fname, im)
