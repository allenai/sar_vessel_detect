import types
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch.nn import functional as F

from xview3.models.frcnn import NoopTransform

class FasterRCNNModelL1(torch.nn.Module):
    def __init__(self, num_classes, num_channels, device, config, image_size=800):
        super(FasterRCNNModelL1, self).__init__()

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

        # Overwrite RPN and ROI loss.
        self.faster_rcnn.rpn.compute_loss = types.MethodType(my_rpn_compute_loss, self.faster_rcnn.rpn)
        torchvision.models.detection.roi_heads.fastrcnn_loss = my_roi_loss

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

def fasterrcnn_resnet101_fpn(num_classes=3, pretrained_backbone=True, trainable_backbone_layers=5, **kwargs):
    backbone = resnet_fpn_backbone(
        backbone_name='resnet101',
        pretrained=pretrained_backbone,
        trainable_layers=trainable_backbone_layers,
    )
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model

def my_rpn_compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
    sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness = objectness.flatten()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    box_loss = (
        F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        )
        / (sampled_inds.numel())
    )

    objectness_probs = torch.sigmoid(objectness[sampled_inds])
    objectness_loss = 5*F.mse_loss(objectness_probs, labels[sampled_inds].float())
    #print('myrpn', objectness_probs, labels[sampled_inds])

    return objectness_loss, box_loss

def my_roi_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    class_probs = F.softmax(class_logits, -1)
    classification_loss = 5*F.mse_loss(class_probs[:, 1], labels.float())
    #print('myroi', class_probs.shape, class_logits, class_probs, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss
