import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator

import random

from xview3.models.frcnn import NoopTransform

class FasterRCNNMultihead(torch.nn.Module):
    def __init__(self, num_classes, num_channels, device, config, image_size=800):
        super(FasterRCNNMultihead, self).__init__()

        image_mean = [float(a) for a in config.get("ImageMean").strip().split(",")]
        image_std = [float(a) for a in config.get("ImageStd").strip().split(",")]
        backbone = config.get("Backbone", fallback="resnet50")
        pretrained = config.getboolean("Pretrained", fallback=True)
        pretrained_backbone = config.getboolean("Pretrained-Backbone", fallback=True)
        trainable_backbone_layers = config.getint("Trainable-Backbone-Layers", fallback=5)
        use_noop_transform = config.getboolean("NoopTransform", fallback=False)

        self.backbone = resnet_fpn_backbone(
            backbone_name=backbone,
            pretrained=True,
            trainable_layers=trainable_backbone_layers,
        )

        # We have max 86 points per 800x800 chip.
        # So here, in case we're using larger image sizes, determine if we need to increase some parameters.
        box_detections_per_img = max(100, 100*image_size*image_size//800//800)
        rpn_pre_nms_top_n_train = max(2000, 2000*image_size*image_size//800//800)
        rpn_post_nms_top_n_train = max(2000, 2000*image_size*image_size//800//800)
        rpn_pre_nms_top_n_test = max(2000, 2000*image_size*image_size//800//800)
        rpn_post_nms_top_n_test = max(2000, 2000*image_size*image_size//800//800)

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

        self.pred_layer = torch.nn.Sequential(
            torch.nn.Conv2d(2048, 512, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(512, eps=1e-3, momentum=0.03),
            torch.nn.Conv2d(512, 512, 4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            #torch.nn.BatchNorm2d(512, eps=1e-3, momentum=0.03),
        )
        self.pred_length = torch.nn.Conv2d(512, 1, 4, stride=2, padding=1)
        self.pred_confidence = torch.nn.Conv2d(512, 3, 4, stride=2, padding=1)
        self.pred_fishing = torch.nn.Conv2d(512, 2, 4, stride=2, padding=1)
        self.pred_vessel = torch.nn.Conv2d(512, 2, 4, stride=2, padding=1)

    def forward(self, *input, **kwargs):
        if self.training:
            images, targets = input
            device = images[0].device

            # Targets for training detector should exclude low-confidence labels.
            detect_targets = []
            for target in targets:
                valid_indices = target['confidence_labels'] >= 1
                new_target = {
                    'boxes': target['boxes'][valid_indices, :],
                    'area': target['area'],
                    'iscrowd': target['iscrowd'],
                    'image_id': target['image_id'],
                }
                if len(new_target['boxes']) == 0:
                    new_target['labels'] = torch.zeros((1,), dtype=torch.int64, device=device)
                else:
                    new_target['labels'] = target['labels'][valid_indices]

                detect_targets.append(new_target)
            detect_loss = self.faster_rcnn.forward(images, detect_targets, **kwargs)

            crops = []
            length_labels = []
            confidence_labels = []
            fishing_labels = []
            vessel_labels = []

            # Get crops of labeled points with slight offsets.
            for i, image in enumerate(images):
                width, height = image.shape[2], image.shape[1]
                image = torch.nn.functional.pad(image, (64, 64, 64, 64))
                for j, (col, row) in enumerate(targets[i]['centers']):
                    col += random.randint(-8, 8)
                    row += random.randint(-8, 8)
                    col = int(torch.clip(col, min=0, max=width))
                    row = int(torch.clip(row, min=0, max=height))
                    crop = image[:, row:row+128, col:col+128]

                    crops.append(crop)
                    length_labels.append(targets[i]['length_labels'][j])
                    confidence_labels.append(targets[i]['confidence_labels'][j])
                    fishing_labels.append(targets[i]['fishing_labels'][j])
                    vessel_labels.append(targets[i]['vessel_labels'][j])

            if len(crops) == 0:
                return detect_loss

            crops = torch.stack(crops, dim=0)
            length_labels = torch.stack(length_labels, dim=0)
            confidence_labels = torch.stack(confidence_labels, dim=0)
            fishing_labels = torch.stack(fishing_labels, dim=0)
            vessel_labels = torch.stack(vessel_labels, dim=0)

            features = self.backbone.body(crops)['3']
            features = self.pred_layer(features)

            length_scores = self.pred_length(features)[:, 0, 0, 0]
            confidence_scores = self.pred_confidence(features)[:, :, 0, 0]
            fishing_scores = self.pred_fishing(features)[:, :, 0, 0]
            vessel_scores = self.pred_vessel(features)[:, :, 0, 0]

            valid_length_indices = length_labels >= 0
            valid_length_labels = length_labels[valid_length_indices]
            valid_length_scores = length_scores[valid_length_indices]
            if len(valid_length_labels) > 0:
                length_loss = torch.div(torch.abs(valid_length_labels - valid_length_scores), valid_length_labels).mean()
            else:
                length_loss = torch.zeros((1,), dtype=torch.float32, device=device)

            def get_ce_loss(labels, scores):
                valid_indices = labels >= 0
                valid_labels = labels[valid_indices]
                valid_scores = scores[valid_indices, :]
                if len(valid_labels) > 0:
                    return torch.nn.functional.cross_entropy(valid_scores, valid_labels)
                else:
                    return torch.zeros((1,), dtype=torch.float32, device=device)

            confidence_loss = get_ce_loss(confidence_labels, confidence_scores)
            fishing_loss = get_ce_loss(fishing_labels, fishing_scores)
            vessel_loss = get_ce_loss(vessel_labels, vessel_scores)

            multihead_loss = length_loss + confidence_loss + fishing_loss + vessel_loss
            detect_loss = sum(detect_loss.values())
            #print('detect={} multihead={} ... length={} conf={} fish={} vessel={}'.format(detect_loss.item(), multihead_loss.item(), length_loss.item(), confidence_loss.item(), fishing_loss.item(), vessel_loss.item()))
            return {'loss': detect_loss + multihead_loss/20}
        else:
            (images,) = input
            device = images[0].device

            outputs = self.faster_rcnn.forward(images, **kwargs)

            # Get crops of predicted points.
            crops = []
            orig_indices = []
            for i, image in enumerate(images):
                width, height = image.shape[2], image.shape[1]
                image = torch.nn.functional.pad(image, (64, 64, 64, 64))

                n = len(outputs[i]['boxes'])
                outputs[i]['labels'] = torch.zeros((n,), dtype=torch.int64, device=device)
                outputs[i]['lengths'] = torch.zeros((n,), dtype=torch.float32, device=device)
                outputs[i]['fishing_scores'] = torch.zeros((n,), dtype=torch.float32, device=device)
                outputs[i]['vessel_scores'] = torch.zeros((n,), dtype=torch.float32, device=device)

                for j, (x1, y1, x2, y2) in enumerate(outputs[i]['boxes']):
                    col = int(torch.clip((x1 + x2)/2, min=0, max=width))
                    row = int(torch.clip((y1 + y2)/2, min=0, max=height))
                    crop = image[:, row:row+128, col:col+128]
                    crops.append(crop)
                    orig_indices.append((i, j))

            if len(crops) == 0:
                return outputs

            crops = torch.stack(crops, dim=0)
            features = self.backbone.body(crops)['3']
            features = self.pred_layer(features)
            length_scores = self.pred_length(features)[:, 0, 0, 0]
            fishing_scores = self.pred_fishing(features)[:, :, 0, 0]
            vessel_scores = self.pred_vessel(features)[:, :, 0, 0]

            fishing_probs = torch.nn.functional.softmax(fishing_scores, dim=1)
            vessel_probs = torch.nn.functional.softmax(vessel_scores, dim=1)

            for idx, (i, j) in enumerate(orig_indices):
                outputs[i]['lengths'][j] = length_scores[idx]
                #print(vessel_scores[idx, :], vessel_scores[idx, :].argmax(), fishing_scores[idx, :], fishing_scores[idx, :].argmax())
                if vessel_scores[idx, :].argmax() == 0:
                    outputs[i]['labels'][j] = 3
                elif fishing_scores[idx, :].argmax() == 0:
                    outputs[i]['labels'][j] = 2
                else:
                    outputs[i]['labels'][j] = 1
                outputs[i]['fishing_scores'][j] = fishing_probs[idx, 1]
                outputs[i]['vessel_scores'][j] = vessel_probs[idx, 1]

            return outputs
