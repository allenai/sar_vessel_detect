import types
import torch
from torch.nn import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import boxes as box_ops
import types

from xview3.models.frcnn import NoopTransform, fasterrcnn_resnet101_fpn

# This is like frcnn_soft_labels but we do the soft labels for the RPN too, not just ROI head.

class FasterRCNNSofterLabels(torch.nn.Module):
    def __init__(self, num_classes, num_channels, device, config, image_size=800):
        super(FasterRCNNSofterLabels, self).__init__()

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
        self.faster_rcnn.rpn.assign_targets_to_anchors = types.MethodType(assign_targets_to_anchors, self.faster_rcnn.rpn)

        self.faster_rcnn.roi_heads.assign_targets_to_proposals = types.MethodType(assign_targets_to_proposals, self.faster_rcnn.roi_heads)
        self.faster_rcnn.roi_heads.select_training_samples = types.MethodType(select_training_samples, self.faster_rcnn.roi_heads)

    def forward(self, *input, **kwargs):
        out = self.faster_rcnn.forward(*input, **kwargs)
        return out

# ** RPN UPDATES **

def assign_targets_to_anchors(self, anchors, targets):
    labels = []
    matched_gt_boxes = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        gt_boxes = targets_per_image["boxes"]

        if gt_boxes.numel() == 0:
            # Background image (negative example)
            device = anchors_per_image.device
            matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
            labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
        else:
            match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            labels_per_image = torch.minimum(labels_per_image, targets_per_image["score_labels"][matched_idxs.clamp(min=0)])

            # Background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0.0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1.0

            # also discard low confidence or <1.0 score
            inds_to_discard = (targets_per_image["score_labels"][matched_idxs.clamp(min=0)] < 1) & (matched_idxs >= 0)
            labels_per_image[inds_to_discard] = -1.0
            inds_to_discard = (targets_per_image["confidence_labels"][matched_idxs.clamp(min=0)] == 0) & (matched_idxs >= 0)
            labels_per_image[inds_to_discard] = -1.0

        labels.append(labels_per_image)
        matched_gt_boxes.append(matched_gt_boxes_per_image)
    return labels, matched_gt_boxes

# ** ROI UPDATES **

def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_confs, gt_scores):
    matched_idxs = []
    labels = []
    for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_confs_in_image, gt_scores_in_image in zip(proposals, gt_boxes, gt_labels, gt_confs, gt_scores):
        if gt_boxes_in_image.numel() == 0:
            # Background image
            device = proposals_in_image.device
            clamped_matched_idxs_in_image = torch.zeros(
                (proposals_in_image.shape[0],), dtype=torch.int64, device=device
            )
            labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
        else:
            #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0

            # Ignore low confidence and <1.0 score labels
            confs_in_image = gt_confs_in_image[clamped_matched_idxs_in_image]
            ignore_inds = (confs_in_image == 0) & (matched_idxs_in_image >= 0)
            labels_in_image[ignore_inds] = -1

            scores_in_image = gt_scores_in_image[clamped_matched_idxs_in_image]
            ignore_inds = (scores_in_image < 1) & (matched_idxs_in_image >= 0)
            labels_in_image[ignore_inds] = -1

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

        matched_idxs.append(clamped_matched_idxs_in_image)
        labels.append(labels_in_image)
    return matched_idxs, labels

def select_training_samples(self, proposals, targets):
    self.check_targets(targets)
    assert targets is not None
    dtype = proposals[0].dtype
    device = proposals[0].device

    gt_boxes = [t["boxes"].to(dtype) for t in targets]
    gt_labels = [t["labels"] for t in targets]
    gt_confs = [t["confidence_labels"] for t in targets]
    gt_scores = [t["score_labels"] for t in targets]

    # append ground-truth bboxes to propos
    proposals = self.add_gt_proposals(proposals, gt_boxes)

    # get matching gt indices for each proposal
    matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_confs, gt_scores)
    # sample a fixed proportion of positive-negative proposals
    sampled_inds = self.subsample(labels)
    matched_gt_boxes = []
    num_images = len(proposals)
    for img_id in range(num_images):
        img_sampled_inds = sampled_inds[img_id]
        proposals[img_id] = proposals[img_id][img_sampled_inds]
        labels[img_id] = labels[img_id][img_sampled_inds]
        matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

        gt_boxes_in_image = gt_boxes[img_id]
        if gt_boxes_in_image.numel() == 0:
            gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
        matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

    regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
    return proposals, matched_idxs, labels, regression_targets
