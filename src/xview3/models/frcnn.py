import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator

class FasterRCNNModel(torch.nn.Module):
    """
    Baseline model class for xView3 reference implementation.
    Wraps torchvision faster-rcnn, updates initial layer to handle
    man arbitrary number of input channels.
    """

    def __init__(self, num_classes, num_channels, device, config, image_size=800, weights=None):
        super(FasterRCNNModel, self).__init__()

        image_mean = [float(a) for a in config.get("ImageMean").strip().split(",")]
        image_std = [float(a) for a in config.get("ImageStd").strip().split(",")]
        backbone = config.get("Backbone", fallback="resnet50")
        pretrained = config.getboolean("Pretrained", fallback=True)
        pretrained_backbone = config.getboolean("Pretrained-Backbone", fallback=True)
        trainable_backbone_layers = config.getint("Trainable-Backbone-Layers", fallback=5)
        use_noop_transform = config.getboolean("NoopTransform", fallback=False)

        # Load in a backbone, with capability to be pretrained on COCO
        if backbone == 'resnet50':
            # We have max 86 points per 800x800 chip.
            # So here, in case we're using larger image sizes, determine if we need to increase some parameters.
            box_detections_per_img = max(100, 100*image_size*image_size//800//800)
            rpn_pre_nms_top_n_train = max(2000, 2000*image_size*image_size//800//800)
            rpn_post_nms_top_n_train = max(2000, 2000*image_size*image_size//800//800)
            rpn_pre_nms_top_n_test = max(2000, 2000*image_size*image_size//800//800)
            rpn_post_nms_top_n_test = max(2000, 2000*image_size*image_size//800//800)

            self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
                image_mean=image_mean,
                image_std=image_std,
                min_size=image_size,
                max_size=image_size,
                pretrained_backbone=pretrained_backbone,
                trainable_backbone_layers=trainable_backbone_layers,
                box_detections_per_img=box_detections_per_img,
                rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            )
        elif backbone == 'resnet101':
            self.faster_rcnn = fasterrcnn_resnet101_fpn(
                num_classes=num_classes,
                pretrained_backbone=pretrained_backbone,
                trainable_backbone_layers=trainable_backbone_layers,
                image_mean=image_mean,
                image_std=image_std,
                min_size=image_size,
                max_size=image_size
            )
        elif backbone == 'mobilenet-320':
            self.faster_rcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                pretrained=pretrained,
                image_mean=image_mean,
                image_std=image_std,
                min_size=image_size,
                max_size=image_size,
                pretrained_backbone=pretrained_backbone,
                trainable_backbone_layers=trainable_backbone_layers
            )
        else:
            raise Exception("Please pass in a valid backbone argument: resnet50, resnet101, mobilenet-320")

        # load in pretrained weights if specified, making sure to ignore the roi_head keys
        if weights:
            weights = torch.load(weights)
            del_list = ['roi_heads.box_predictor.cls_score.weight', 'roi_heads.box_predictor.cls_score.bias', 
                        'roi_heads.box_predictor.bbox_pred.weight', 'roi_heads.box_predictor.bbox_pred.bias']
            [weights.pop(key) for key in del_list]
            self.faster_rcnn.load_state_dict(weights, strict=False)  

        # replace the anchor generator with custom sizes and aspect ratios
        # TODO: this number of sizes and ratios is hardcoded here because FRCNN's desired format is odd, clean up
        anchor_sizes = config.get("AnchorSizes", fallback=None)
        anchor_ratios = config.get("AnchorRatios", fallback=None)
        if anchor_sizes:
            s = [int(s) for s in anchor_sizes.strip().split(",")]
            r = [float(r) for r in anchor_ratios.strip().split(",")]
            sizes = ((s[0],), (s[1],), (s[2],), (s[3],), (s[4],))
            ratios = ((r[0], r[1], r[2]),) * len(sizes)
            anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
            self.faster_rcnn.rpn.anchor_generator = anchor_generator

        # replace the classifier with a new one for user-defined num_classes
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        if use_noop_transform:
            self.faster_rcnn.transform = NoopTransform()

        print(f"Using {num_channels} channels for input layer...")
        self.num_channels = num_channels
        if num_channels > 3:
            # Adjusting initial layer to handle arbitrary number of inputchannels
            self.faster_rcnn.backbone.body.conv1 = torch.nn.Conv2d(
                num_channels,
                self.faster_rcnn.backbone.body.conv1.out_channels,
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

        out = self.faster_rcnn.forward(*input, **kwargs)
        return out


class NoopTransform(torch.nn.Module):
    def __init__(self):
        super(NoopTransform, self).__init__()

        self.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800,
            max_size=800,
            image_mean=[],
            image_std=[],
        )

    def forward(self, images, targets):
        images = self.transform.batch_images(images, size_divisible=32)
        image_sizes = [(image.shape[1], image.shape[2]) for image in images]
        image_list = torchvision.models.detection.image_list.ImageList(images, image_sizes)
        return image_list, targets

    def postprocess(self, detections, image_sizes, orig_sizes):
        return detections


def fasterrcnn_resnet101_fpn(num_classes=3, pretrained_backbone=True, trainable_backbone_layers=5, **kwargs):
    backbone = resnet_fpn_backbone(
        backbone_name='resnet101',
        pretrained=pretrained_backbone,
        trainable_layers=trainable_backbone_layers,
    )
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model
