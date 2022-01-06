import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNet

class RetinaNetModel(torch.nn.Module):
    """
    RetinaNet model class for xView3 reference implementation.
    Wraps torchvision retinanet, updates initial layer to handle
    man arbitrary number of input channels.
    """

    def __init__(self, num_classes, num_channels, device, config, image_size=800):
        super(RetinaNetModel, self).__init__()

        image_mean = [float(a) for a in config.get("ImageMean").strip().split(",")]
        image_std = [float(a) for a in config.get("ImageStd").strip().split(",")]
        pretrained = config.getboolean("Pretrained")
        pretrained_backbone = config.getboolean("Pretrained-Backbone")
        trainable_backbone_layers = config.getint("Trainable-Backbone-Layers")

        # load the model pre-trained on COCO
        self.retinanet = torchvision.models.detection.retinanet_resnet50_fpn(
            pretrained=pretrained,
            image_mean=image_mean,
            image_std=image_std,
            min_size=image_size,
            max_size=image_size,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers
        )

        print(f"Using {num_channels} channels for input layer...")
        self.num_channels = num_channels
        if num_channels > 3:
            # Adjusting initial layer to handle arbitrary number of inputchannels
            self.retinanet.backbone.body.conv1 = torch.nn.Conv2d(
                num_channels,
                self.retinanet.backbone.body.conv1.out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, *input, **kwargs):
        if self.num_channels < 3:
            # Copy the first channel until we have three channel input.
            input = list(input)
            new_imgs = []
            for img in input[0]:
                l = [img]
                for _ in range(3-self.num_channels):
                    l.append(img[0:1])
                new_imgs.append(torch.cat(l, dim=0))
            input[0] = new_imgs
            input = tuple(input)

        out = self.retinanet.forward(*input, **kwargs)
        return out
