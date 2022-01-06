import numpy as np
import torch

class DefaultNormalize(object):
    '''
    Apply the normalization from reference model.
    '''

    def __init__(self, info):
        self.channels = info['channels']

    def __call__(self, image, targets):
        for ch, channel in enumerate(self.channels):
            if channel == "wind_direction":
                image[ch][image[ch] < 0] = np.random.randint(0, 360, size=1).item()
                image[ch][image[ch] > 360] = np.random.randint(0, 360, size=1).item()
                image[ch] = image[ch] - 180
            if channel == "wind_speed":
                image[ch][image[ch] < 0] = 0
                image[ch][image[ch] > 100] = 100
            if channel in ["vh", "vv", "vh_other"]:
                image[ch][image[ch] < -50] = -50
        return image, targets

class CustomNormalize(object):
    '''
    Normalization that retains invalid pixel values and incorporates bathymetry.
    Also makes values close to [-1, 1].
    '''

    def __init__(self, info):
        self.channels = info['channels']

    def __call__(self, image, targets):
        for ch, channel in enumerate(self.channels):
            if channel == "wind_direction":
                image[ch][image[ch] < 0] = np.random.randint(0, 360, size=1).item()
                image[ch][image[ch] > 360] = np.random.randint(0, 360, size=1).item()
                image[ch] = (image[ch] - 180)/360
            if channel == "wind_speed":
                image[ch][image[ch] < 0] = 0
                image[ch][image[ch] > 100] = 100
            if channel in ["vh", "vv", "vh_other"]:
                image[ch][(image[ch] < -50) & (image[ch] > -30000)] = -50
                image[ch][image[ch] < -100] = -100
                image[ch] = image[ch]/50
            if channel == "bathymetry":
                image[ch][(image[ch] < -5000) & (image[ch] > -30000)] = -5000
                image[ch][image[ch] < -10000] = -10000
                image[ch] = image[ch]/5000
        return image, targets

class CustomNormalize2(object):
    '''
    Like CustomNormalize, but doesn't separate out invalid pixels, and output
    values are in [0, 1].
    '''

    def __init__(self, info):
        self.channels = info['channels']

    def __call__(self, image, targets):
        for ch, channel in enumerate(self.channels):
            if channel == "wind_direction":
                image[ch][image[ch] < 0] = np.random.randint(0, 360, size=1).item()
                image[ch][image[ch] > 360] = np.random.randint(0, 360, size=1).item()
                image[ch] = (image[ch] - 180)/360
            if channel == "wind_speed":
                image[ch][image[ch] < 0] = 0
                image[ch][image[ch] > 100] = 100
                image[ch] = image[ch]/100
            if channel in ["vh", "vv", "vh_other"]:
                image[ch, :, :] = (torch.clip(image[ch, :, :], min=-50, max=20)+50)/70
            if channel == "bathymetry":
                image[ch, :, :] = (torch.clip(image[ch, :, :], min=-6000, max=2000)+6000)/8000
        return image, targets

class CustomNormalize3(object):
    '''
    Like CustomNormalize2, but use sigmoid for bathymetry.
    '''

    def __init__(self, info):
        self.channels = info['channels']

    def __call__(self, image, targets):
        for ch, channel in enumerate(self.channels):
            if channel == "wind_direction":
                image[ch][image[ch] < 0] = np.random.randint(0, 360, size=1).item()
                image[ch][image[ch] > 360] = np.random.randint(0, 360, size=1).item()
                image[ch] = (image[ch] - 180)/360
            if channel == "wind_speed":
                image[ch][image[ch] < 0] = 0
                image[ch][image[ch] > 100] = 100
                image[ch] = image[ch]/100
            if channel in ["vh", "vv", "vh_other"]:
                image[ch, :, :] = (torch.clip(image[ch, :, :], min=-50, max=20)+50)/70
            if channel == "bathymetry":
                image[ch, :, :] = (np.cbrt(torch.clip(image[ch, :, :], min=-6000, max=2000))+18.2)/31
        return image, targets

class MinMaxNormalize(object):
    '''
    Make values between [0, 1] with per-image normalization.
    '''

    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        for ch in range(image.shape[0]):
            image[ch] = (image[ch] - image[ch].min()) / (image[ch].max() - image[ch].min())
        return image, targets
