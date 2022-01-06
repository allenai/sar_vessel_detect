from xview3.transforms.augment import FlipLR, Crop32, Crop224, Crop800, Crop1200, Rotate, Noise, Jitter, Jitter2, FlipUD, Bucket
from xview3.transforms.normalize import DefaultNormalize, CustomNormalize, CustomNormalize2, CustomNormalize3, MinMaxNormalize

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets):
        for transform in self.transforms:
            image, targets = transform(image, targets)
        return image, targets

def get_transforms(names, info):
    import xview3.transforms as xview_transforms
    transforms = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        transform_cls = getattr(xview_transforms, name)
        transform = transform_cls(info)
        transforms.append(transform)

    if transforms:
        return Compose(transforms)
    else:
        return None
