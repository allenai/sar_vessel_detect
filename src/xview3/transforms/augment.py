import math
import random
import torch
import torchvision

class FlipLR(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            targets['centers'][:, 0] = image.shape[2] - targets['centers'][:, 0]
            targets['boxes'] = torch.stack([
                image.shape[2] - targets['boxes'][:, 2],
                targets['boxes'][:, 1],
                image.shape[2] - targets['boxes'][:, 0],
                targets['boxes'][:, 3],
            ], dim=1)
        return image, targets

class FlipUD(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            targets['centers'][:, 1] = image.shape[1] - targets['centers'][:, 1]
            targets['boxes'] = torch.stack([
                targets['boxes'][:, 0],
                image.shape[1] - targets['boxes'][:, 3],
                targets['boxes'][:, 2],
                image.shape[1] - targets['boxes'][:, 1],
            ], dim=1)
        return image, targets

class Crop(object):
    def __init__(self, info, amount):
        self.amount = amount

    def __call__(self, image, targets):
        target_size = image.shape[1] - self.amount

        # Assume vh is first channel.
        x_valid = image[0, :, :].amax(axis=1)
        y_valid = image[0, :, :].amax(axis=0)
        if len(x_valid) > 0 and len(y_valid) > 0:
            sx = torch.nonzero(x_valid)[0]
            ex = torch.nonzero(x_valid)[-1]+1
            sy = torch.nonzero(y_valid)[0]
            ey = torch.nonzero(y_valid)[-1]+1
        else:
            sx, ex, sy, ey = 0, image.shape[2], 0, image.shape[1]

        sx = max(0, sx-target_size)
        ex = min(image.shape[2], ex+target_size)
        sy = max(0, sy-target_size)
        ey = min(image.shape[1], ey+target_size)

        left = random.randint(sx, ex-target_size)
        right = left + target_size
        top = random.randint(sy, ey-target_size)
        bottom = top + target_size
        image = image[:, top:bottom, left:right]

        if len(targets['boxes']) == 0:
            return image, targets

        valid_indices = (targets['centers'][:, 0] > left) & (targets['centers'][:, 0] < right) & (targets['centers'][:, 1] > top) & (targets['centers'][:, 1] < bottom)
        targets['centers'] = targets['centers'][valid_indices, :].contiguous()
        targets['boxes'] = targets['boxes'][valid_indices, :].contiguous()
        targets['labels'] = targets['labels'][valid_indices].contiguous()
        targets['length_labels'] = targets['length_labels'][valid_indices].contiguous()
        targets['confidence_labels'] = targets['confidence_labels'][valid_indices].contiguous()
        targets['fishing_labels'] = targets['fishing_labels'][valid_indices].contiguous()
        targets['vessel_labels'] = targets['vessel_labels'][valid_indices].contiguous()
        targets['score_labels'] = targets['score_labels'][valid_indices].contiguous()

        targets['centers'][:, 0] -= left
        targets['centers'][:, 1] -= top
        targets['boxes'][:, 0] -= left
        targets['boxes'][:, 1] -= top
        targets['boxes'][:, 2] -= left
        targets['boxes'][:, 3] -= top

        # Weird special case.
        if len(targets['boxes']) == 0:
            targets['labels'] = torch.zeros((1,), dtype=torch.int64)

        return image, targets

class Crop32(Crop):
    def __init__(self, info):
        super(Crop32, self).__init__(info, amount=32)

class Crop224(Crop):
    def __init__(self, info):
        super(Crop224, self).__init__(info, amount=224)

class Crop800(Crop):
    def __init__(self, info):
        super(Crop800, self).__init__(info, amount=800)

class Crop1200(Crop):
    def __init__(self, info):
        super(Crop1200, self).__init__(info, amount=1200)

class Rotate(object):
    def __init__(self, info):
        self.bbox_size = info['bbox_size']

    def __call__(self, image, targets):
        angle_deg = random.randint(0, 359)
        angle_rad = angle_deg * math.pi / 180
        image = torchvision.transforms.functional.rotate(image, angle_deg)

        if len(targets['boxes']) == 0:
            return image, targets

        im_center = (image.shape[2]//2, image.shape[1]//2)
        # Subtract center.
        centers = torch.stack([
            targets['centers'][:, 0] - im_center[0],
            targets['centers'][:, 1] - im_center[1],
        ], dim=1)
        # Rotate around origin.
        centers = torch.stack([
            math.sin(angle_rad)*centers[:, 1] + math.cos(angle_rad)*centers[:, 0],
            math.cos(angle_rad)*centers[:, 1] - math.sin(angle_rad)*centers[:, 0],
        ], dim=1)
        # Add back the center.
        centers = torch.stack([
            centers[:, 0] + im_center[0],
            centers[:, 1] + im_center[1],
        ], dim=1)
        # Prune ones outside image window.
        valid_indices = (centers[:, 0] > 0) & (centers[:, 0] < image.shape[2]) & (centers[:, 1] > 0) & (centers[:, 1] < image.shape[1])
        centers = centers[valid_indices, :].contiguous()
        targets['centers'] = centers
        targets['boxes'] = torch.stack([
            centers[:, 0] - self.bbox_size,
            centers[:, 1] - self.bbox_size,
            centers[:, 0] + self.bbox_size,
            centers[:, 1] + self.bbox_size,
        ], dim=1)
        targets['labels'] = targets['labels'][valid_indices].contiguous()
        targets['length_labels'] = targets['length_labels'][valid_indices].contiguous()
        targets['confidence_labels'] = targets['confidence_labels'][valid_indices].contiguous()
        targets['fishing_labels'] = targets['fishing_labels'][valid_indices].contiguous()
        targets['vessel_labels'] = targets['vessel_labels'][valid_indices].contiguous()
        targets['score_labels'] = targets['score_labels'][valid_indices].contiguous()

        # Weird special case.
        if len(targets['boxes']) == 0:
            targets['labels'] = torch.zeros((1,), dtype=torch.int64)

        return image, targets

class Noise(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        image = image + 0.1*torch.randn(image.shape)
        image = torch.clip(image, min=0, max=1)
        return image, targets

class Jitter(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        jitter = 0.4*(torch.rand(image.shape[0])-0.5)
        image = image + jitter[:, None, None]
        image = torch.clip(image, min=0, max=1)
        return image, targets

class Jitter2(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        jitter = 0.1*(torch.rand(image.shape[0])-0.5)
        image = image + jitter[:, None, None]
        image = torch.clip(image, min=0, max=1)
        return image, targets

class Bucket(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        for channel_idx in [0, 1]:
            buckets = torch.tensor([(i+1)/10 + (random.random()-0.5)/10 for i in range(9)], device=image.device)
            image[channel_idx, :, :] = torch.bucketize(image[channel_idx, :, :], buckets).float()/10
        return image, targets
