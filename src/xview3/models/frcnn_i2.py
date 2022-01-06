import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import boxes as box_ops
import types

import random

from xview3.models.frcnn import NoopTransform

class SimpleBackbone(torch.nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()

        def down_layer(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
                torch.nn.ReLU(inplace=True),
                #torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                #torch.nn.ReLU(inplace=True),
            )

        def up_layer(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
                torch.nn.ReLU(inplace=True),
            )

        self.down1 = down_layer(3, 32) # -> 400x400
        self.down2 = down_layer(32, 64) # -> 200x200
        self.down3 = down_layer(64, 128) # -> 100x100
        self.down4 = down_layer(128, 256) # -> 50x50
        self.down5 = down_layer(256, 512) # -> 25x25
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 32, 3, padding=1),
        )

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        features = self.features(down5)
        return features

class MyBackbone(torch.nn.Module):
    def __init__(self, backbone):
        super(MyBackbone, self).__init__()
        self.backbone = backbone
        self.simple_backbone = SimpleBackbone()
        self.out_channels = 32#self.backbone.out_channels+32

    def forward(self, x):
        my_output = self.backbone(x[:, 0:3, :, :])
        other_features = torch.zeros((x.shape[0], 32, x.shape[2]//32, x.shape[3]//32), dtype=torch.float32, device=x.device)

        valid = (x[:, (3, 6, 9, 12), :, :].amax(dim=[2, 3]) > 0).float()
        lengths = torch.clip(valid.sum(dim=1), min=1)

        for i in range(4):
            other_features += (self.simple_backbone(x[:, 3*(i+1):3*(i+2), :, :]) * valid[:, i, None, None, None])/lengths[:, None, None, None]

        output = {}
        for k, mine in my_output.items():
            transformed_features = torchvision.transforms.functional.resize(other_features, [mine.shape[2], mine.shape[3]])
            output[k] = transformed_features#torch.cat([mine, transformed_features], dim=1)
        return output

class FasterRCNNi2(torch.nn.Module):
    def __init__(self, num_classes, num_channels, device, config, image_size=800):
        super(FasterRCNNi2, self).__init__()

        image_mean = [float(a) for a in config.get("ImageMean").strip().split(",")]
        image_std = [float(a) for a in config.get("ImageStd").strip().split(",")]
        backbone = config.get("Backbone", fallback="resnet50")
        pretrained = config.getboolean("Pretrained", fallback=True)
        pretrained_backbone = config.getboolean("Pretrained-Backbone", fallback=True)
        trainable_backbone_layers = config.getint("Trainable-Backbone-Layers", fallback=5)
        use_noop_transform = config.getboolean("NoopTransform", fallback=False)

        # We have max 86 points per 800x800 chip.
        # So here, in case we're using larger image sizes, determine if we need to increase some parameters.
        box_detections_per_img = max(100, 100*image_size*image_size//800//800)
        rpn_pre_nms_top_n_train = max(2000, 2000*image_size*image_size//800//800)
        rpn_post_nms_top_n_train = max(2000, 2000*image_size*image_size//800//800)
        rpn_pre_nms_top_n_test = max(1000, 1000*image_size*image_size//800//800)
        rpn_post_nms_top_n_test = max(1000, 1000*image_size*image_size//800//800)

        if backbone.startswith('resnet'):
            self.backbone = resnet_fpn_backbone(
                backbone_name=backbone,
                pretrained=pretrained,
                trainable_layers=trainable_backbone_layers,
            )
            self.backbone = MyBackbone(self.backbone)

            self.faster_rcnn = FasterRCNN(
                self.backbone, num_classes,
                image_mean=image_mean,
                image_std=image_std,
                min_size=image_size,
                max_size=image_size,
                box_detections_per_img=box_detections_per_img,
                rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            )

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

        # Overwrite RPN and ROI loss.
        self.faster_rcnn.rpn.assign_targets_to_anchors = types.MethodType(assign_targets_to_anchors, self.faster_rcnn.rpn)
        self.faster_rcnn.rpn.compute_loss = types.MethodType(my_rpn_compute_loss, self.faster_rcnn.rpn)

        self.faster_rcnn.roi_heads.assign_targets_to_proposals = types.MethodType(assign_targets_to_proposals, self.faster_rcnn.roi_heads)
        self.faster_rcnn.roi_heads.forward = types.MethodType(roi_heads_forward, self.faster_rcnn.roi_heads)
        self.faster_rcnn.roi_heads.select_training_samples = types.MethodType(select_training_samples, self.faster_rcnn.roi_heads)

    def forward(self, *input, **kwargs):
        return self.faster_rcnn.forward(*input, **kwargs)

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

            # also discard low confidence
            inds_to_discard = (targets_per_image["confidence_labels"][matched_idxs.clamp(min=0)] == 0) & (matched_idxs >= 0)
            labels_per_image[inds_to_discard] = -1.0

        labels.append(labels_per_image)
        matched_gt_boxes.append(matched_gt_boxes_per_image)
    return labels, matched_gt_boxes

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

    objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])
    #print('rpn', objectness_loss, objectness[sampled_inds], labels[sampled_inds])

    return objectness_loss, box_loss

# ** ROI UPDATES **

def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_confs, gt_scores):
    matched_idxs = []
    labels = []
    scores = []
    for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_confs_in_image, gt_scores_in_image in zip(proposals, gt_boxes, gt_labels, gt_confs, gt_scores):
        if gt_boxes_in_image.numel() == 0:
            # Background image
            device = proposals_in_image.device
            clamped_matched_idxs_in_image = torch.zeros(
                (proposals_in_image.shape[0],), dtype=torch.int64, device=device
            )
            labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            scores_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.float32, device=device)
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

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            scores_in_image = torch.minimum(labels_in_image.float(), gt_scores_in_image[clamped_matched_idxs_in_image])

        matched_idxs.append(clamped_matched_idxs_in_image)
        labels.append(labels_in_image)
        scores.append(scores_in_image)
    return matched_idxs, labels, scores

def my_roi_loss(class_logits, box_regression, labels, regression_targets, scores):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    scores = torch.cat(scores, dim=0)

    fixed_labels = torch.zeros((scores.shape[0], class_logits.shape[1]), device=class_logits.device, dtype=torch.float32)
    fixed_labels[:, 0] = 1 - scores
    fixed_labels[:, 1] = scores
    #classification_loss = F.cross_entropy(class_logits, fixed_labels)
    log_probs = F.log_softmax(class_logits, dim=1)
    raw_ce = -torch.sum(fixed_labels * log_probs, dim=1)
    classification_loss = torch.mean(raw_ce)
    #print('cls loss', classification_loss, fixed_labels[:, 1], torch.softmax(class_logits, dim=1)[:, 1])

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
    matched_idxs, labels, scores = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_confs, gt_scores)
    # sample a fixed proportion of positive-negative proposals
    sampled_inds = self.subsample(labels)
    matched_gt_boxes = []
    num_images = len(proposals)
    for img_id in range(num_images):
        img_sampled_inds = sampled_inds[img_id]
        proposals[img_id] = proposals[img_id][img_sampled_inds]
        labels[img_id] = labels[img_id][img_sampled_inds]
        scores[img_id] = scores[img_id][img_sampled_inds]
        matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

        gt_boxes_in_image = gt_boxes[img_id]
        if gt_boxes_in_image.numel() == 0:
            gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
        matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

    regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
    return proposals, matched_idxs, labels, regression_targets, scores

def roi_heads_forward(self, features, proposals, image_shapes, targets=None,):
    """
    Args:
        features (List[Tensor])
        proposals (List[Tensor[N, 4]])
        image_shapes (List[Tuple[H, W]])
        targets (List[Dict])
    """
    if targets is not None:
        for t in targets:
            # TODO: https://github.com/pytorch/pytorch/issues/26731
            floating_point_types = (torch.float, torch.double, torch.half)
            assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
            assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
            if self.has_keypoint():
                assert t["keypoints"].dtype == torch.float32, "target keypoints must of float type"

    if self.training:
        proposals, matched_idxs, labels, regression_targets, scores = self.select_training_samples(proposals, targets)
    else:
        labels = None
        regression_targets = None
        matched_idxs = None

    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    losses = {}
    if self.training:
        assert labels is not None and regression_targets is not None
        loss_classifier, loss_box_reg = my_roi_loss(class_logits, box_regression, labels, regression_targets, scores)
        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    else:
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

    if self.has_mask():
        mask_proposals = [p["boxes"] for p in result]
        if self.training:
            assert matched_idxs is not None
            # during training, only focus on positive boxes
            num_images = len(proposals)
            mask_proposals = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.where(labels[img_id] > 0)[0]
                mask_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            pos_matched_idxs = None

        if self.mask_roi_pool is not None:
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)
        else:
            raise Exception("Expected mask_roi_pool to be not None")

        loss_mask = {}
        if self.training:
            assert targets is not None
            assert pos_matched_idxs is not None
            assert mask_logits is not None

            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
            loss_mask = {"loss_mask": rcnn_loss_mask}
        else:
            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

        losses.update(loss_mask)

    # keep none checks in if conditional so torchscript will conditionally
    # compile each branch
    if (
        self.keypoint_roi_pool is not None
        and self.keypoint_head is not None
        and self.keypoint_predictor is not None
    ):
        keypoint_proposals = [p["boxes"] for p in result]
        if self.training:
            # during training, only focus on positive boxes
            num_images = len(proposals)
            keypoint_proposals = []
            pos_matched_idxs = []
            assert matched_idxs is not None
            for img_id in range(num_images):
                pos = torch.where(labels[img_id] > 0)[0]
                keypoint_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            pos_matched_idxs = None

        keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
        keypoint_features = self.keypoint_head(keypoint_features)
        keypoint_logits = self.keypoint_predictor(keypoint_features)

        loss_keypoint = {}
        if self.training:
            assert targets is not None
            assert pos_matched_idxs is not None

            gt_keypoints = [t["keypoints"] for t in targets]
            rcnn_loss_keypoint = keypointrcnn_loss(
                keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
            )
            loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
        else:
            assert keypoint_logits is not None
            assert keypoint_proposals is not None

            keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
            for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                r["keypoints"] = keypoint_prob
                r["keypoints_scores"] = kps

        losses.update(loss_keypoint)

    return result, losses
