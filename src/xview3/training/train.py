import configparser
import numpy as np
import math
import os
import pandas as pd
import sys
import random
import time
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torch.cuda.amp
import torchvision

sys.path.insert(1, '/home/xview3/src') # use an appropriate path if not in the docker volume

from xview3.processing.dataloader import SARDataset
import xview3.training.utils
import xview3.training.ema
import xview3.models
import xview3.transforms
import xview3.infer.inference_chip
import xview3.eval.prune
import xview3.eval.metric

def main(config):
    # data params
    chips_path = config.get("data", "ChipsPath")
    train_scene_path = config.get("data", "TrainScenePath")
    val_scene_path = config.get("data", "ValScenePath")
    custom_annotation_path = config.get("data", "CustomAnnotationPath", fallback=None)
    channels = config.get("data", "Channels").strip().split(",")
    num_loader_workers = config.getint("data", "LoaderWorkers")
    skip_low_confidence = config.getboolean("data", "SkipLowConfidence", fallback=False)
    class_map = config.get("data", "ClassMap", fallback=None)
    transform_names = config.get("data", "Transforms").split(",")
    train_transform_names = config.get("data", "TrainTransforms", fallback="").split(",")
    background_frac = config.getfloat("data", "BackgroundFrac", fallback=None)
    use_box_labels = config.getboolean("data", "UseBoxLabels", fallback=False)
    bbox_size = config.getint("data", "BboxSize", fallback=5)
    clip_boxes = config.getboolean("data", "ClipBoxes", fallback=False)
    all_chips = config.getboolean("data", "AllChips", fallback=False)
    val_all_chips = config.getboolean("data", "ValAllChips", fallback=False)
    near_shore_only = config.getboolean("data", "NearShoreOnly", fallback=False)
    span = config.getint("data", "Span", fallback=1)
    geosplit = config.get("data", "GeoSplit", fallback=None)
    use_geo_balanced_sampler = config.getboolean("data", "GeoBalancedSampler", fallback=False)
    use_bg_balanced_sampler = config.getboolean("data", "BGBalancedSampler", fallback=False)
    bg_sampler_incl_low_score = config.getboolean("data", "BGSamplerInclLowScore", fallback=True)
    bg_sampler_only_val_bg = config.getboolean("data", "BGSamplerOnlyValBG", fallback=False)
    shore_root = config.get("data", "ShoreRoot", fallback="/xview3/shoreline/validation/")
    histogram_hide_prob = config.getfloat("data", "HistogramHideProb", fallback=None)
    chip_list = config.get("data", "ChipList", fallback=None)
    i2 = config.getboolean("data", "I2", fallback=False)

    if class_map is not None:
        class_map = [int(cls) for cls in class_map.split(',')]

    # model params
    is_distributed = config.getboolean("training", "IsDistributed", fallback=False)
    batch_size = config.getint("training", "BatchSize")
    effective_batch_size = config.getint("training", "EffectiveBatchSize", fallback=None)
    model_name = config.get("training", "Model")
    num_epochs = int(config["training"]["NumberEpochs"])
    save_path = config["training"]["SavePath"]
    optimizer_mode = config.get("training", "Optimizer", fallback="reference")
    learning_rate = config.getfloat("training", "LearningRate", fallback=0.005)
    half_enabled = config.getboolean("training", "Half", fallback=False)
    patience = config.getint("training", "Patience", fallback=10)
    summary_frequency = config.getint("training", "SummaryFrequency", fallback=8192)
    restore_path = config.get("training", "RestorePath", fallback=None)
    random_resize = config.getfloat("training", "RandomResize", fallback=None)
    freeze_weights = config.get("training", "FreezeWeights", fallback=None)
    freeze_examples = config.getint("training", "FreezeExamples", fallback=None)
    ema_factor = config.getfloat("training", "EMA", fallback=None)

    transform_info = {
        'channels': channels,
        'bbox_size': bbox_size,
    }
    transforms = xview3.transforms.get_transforms(transform_names, transform_info)
    train_transforms = xview3.transforms.get_transforms(transform_names + train_transform_names, transform_info)

    # same place, temp for testing
    train_data = SARDataset(
        chips_path=chips_path,
        scene_path=train_scene_path,
        transforms=train_transforms,
        channels=channels,
        skip_low_confidence=skip_low_confidence,
        class_map=class_map,
        background_frac=background_frac,
        use_box_labels=use_box_labels,
        bbox_size=bbox_size,
        clip_boxes=clip_boxes,
        all_chips=all_chips,
        near_shore_only=near_shore_only,
        span=span,
        custom_annotation_path=custom_annotation_path,
        geosplit=geosplit,
        histogram_hide_prob=histogram_hide_prob,
        chip_list=chip_list,
        i2=i2,
    )

    val_data = SARDataset(
        chips_path=chips_path,
        scene_path=val_scene_path,
        transforms=transforms,
        channels=channels,
        skip_low_confidence=skip_low_confidence,
        class_map=class_map,
        background_frac=None,
        use_box_labels=use_box_labels,
        bbox_size=bbox_size,
        clip_boxes=clip_boxes,
        all_chips=val_all_chips,
        custom_annotation_path=custom_annotation_path,
        chip_list=chip_list,
        i2=i2,
    )

    # train on the GPU or on the CPU, if a GPU is not available
    # os.environ['CUDA_VISIBLE_DEVICES']="3"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('training on device {}'.format(device))

    # define training and validation data loaders

    if is_distributed:
        torch.distributed.init_process_group(
            backend="nccl", world_size=torch.cuda.device_count()
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        if use_geo_balanced_sampler:
            train_sampler = train_data.get_geo_balanced_sampler()
        elif use_bg_balanced_sampler:
            train_sampler = train_data.get_bg_balanced_sampler(background_frac=background_frac, incl_low_score=bg_sampler_incl_low_score, only_val_bg=bg_sampler_only_val_bg)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_data)

        val_sampler = torch.utils.data.SequentialSampler(val_data)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_loader_workers,
        collate_fn=xview3.training.utils.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_loader_workers,
        collate_fn=xview3.training.utils.collate_fn,
    )

    # instantiate model with a number of classes
    model_cls = xview3.models.models[model_name]
    image_size = train_data[0][0].shape[1]
    print('image_size={}'.format(image_size))
    model = model_cls(
        num_classes=4,
        num_channels=len(train_data.channels),
        device=device,
        config=config["training"],
        image_size=image_size,
    )

    # Restore saved model if requested.
    if restore_path:
        print('Restoring model from', restore_path)
        state_dict = torch.load(restore_path)
        model.load_state_dict(state_dict)

    if ema_factor:
        print('creating EMA model')
        model = xview3.training.ema.EMA(model, decay=ema_factor)

    # move model to the correct device
    model.to(device)

    if is_distributed:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[a for a in range(torch.cuda.device_count())],
            broadcast_buffers=False,
        )
        # TODO: Fix DataParallel
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_mode == "adam":
        optimizer = torch.optim.Adam(params, lr=learning_rate)
    elif optimizer_mode == "reference":
        if model_name == 'yolov5':
            g0, g1, g2 = [], [], []  # optimizer parameter groups
            for v in model.model.modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                    g2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                    g0.append(v.weight)
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                    g1.append(v.weight)

            optimizer = torch.optim.SGD(g0, lr=learning_rate, momentum=0.937, nesterov=True)
            optimizer.add_param_group({'params': g1, 'weight_decay': 0.0005})
            optimizer.add_param_group({'params': g2})
        else:
            optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, min_lr=1e-5, cooldown=5)
    warmup_iters = 32768 // batch_size
    warmup_lr_scheduler = xview3.training.utils.warmup_lr_scheduler(optimizer, warmup_iters, 1.0/warmup_iters)

    scaler = torch.cuda.amp.GradScaler(enabled=half_enabled)

    # TensorBoard logging
    cur_iterations = 0
    t00 = time.time()
    summary_writer = torch.utils.tensorboard.SummaryWriter(os.path.join(save_path, 'logs'))
    summary_iters = summary_frequency // batch_size
    summary_epoch = 0
    summary_save_freq = 5
    summary_prev_time = time.time()
    train_losses = []

    if effective_batch_size:
        accumulate_freq = effective_batch_size // batch_size
    else:
        accumulate_freq = 1

    # Evaluation-related variables.
    best_score = 0.0
    gt = pd.read_csv(os.path.join(chips_path, 'chip_annotations.csv'))
    gt = gt[gt.scene_id.isin(val_data.scenes)]
    gt = gt.reset_index()
    gt_incl_low = gt
    gt = gt[gt.confidence.isin(["HIGH", "MEDIUM"])]
    gt = gt.reset_index(drop=True)

    if freeze_weights:
        for name, param in model.named_parameters():
            if not name.startswith(freeze_weights):
                print('not freezing', name)
                continue
            print('freezing', name)
            param.requires_grad = False

    model.train()
    for epoch in range(num_epochs):
        print('begin epoch {}'.format(epoch))

        model.train()
        optimizer.zero_grad()

        for images, targets in train_loader:
            cur_iterations += 1

            #print((time.time()-t00)/cur_iterations)

            if freeze_examples and cur_iterations >= freeze_examples // batch_size:
                print('unfreezing!')
                for name, param in model.named_parameters():
                    if not name.startswith(freeze_weights):
                        continue
                    param.requires_grad = True
                freeze_examples = None

            images = list(image.to(device) for image in images)
            targets = [
                {k: v.to(device) for k, v in t.items() if not isinstance(v, str)}
                for t in targets
            ]

            if random_resize is not None:
                resize_factor = random.uniform(-random_resize, random_resize)
                orig_size = images[0].shape[1]
                target_size = 32*int((1+resize_factor)*orig_size/32)
                resize_factor = target_size / orig_size

                images = [torchvision.transforms.functional.resize(image, size=[target_size, target_size]) for image in images]
                #print('pre', orig_size, target_size, images[0].shape, targets[0]['centers'], targets[0]['boxes'])
                for target in targets:
                    target['centers'] *= resize_factor
                    if use_box_labels:
                        target['boxes'] *= resize_factor
                    else:
                        target['boxes'] = torch.stack([
                            target['centers'][:, 0] - bbox_size,
                            target['centers'][:, 1] - bbox_size,
                            target['centers'][:, 0] + bbox_size,
                            target['centers'][:, 1] + bbox_size,
                        ], dim=1)
                    if clip_boxes:
                        target['boxes'] = torch.stack([
                            torch.clip(target['boxes'][:, 0], min=0, max=target_size),
                            torch.clip(target['boxes'][:, 1], min=0, max=target_size),
                            torch.clip(target['boxes'][:, 2], min=0, max=target_size),
                            torch.clip(target['boxes'][:, 3], min=0, max=target_size),
                        ], dim=1)
                #print('post', orig_size, target_size, images[0].shape, targets[0]['centers'], targets[0]['boxes'])

            with torch.cuda.amp.autocast(enabled=half_enabled):
                loss_dict = model(images, targets)
                #print(loss_dict)
                losses = sum( (loss * float(config.get("training", "Coeff"+str(name), fallback=1.0))) for name, loss in loss_dict.items())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = xview3.training.utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            scaler.scale(losses).backward()

            if cur_iterations == 1 or cur_iterations%accumulate_freq == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema_factor:
                    model.update(summary_epoch)

                if model_name == 'yolov5':
                    model.ema.update(model.model)

            train_losses.append(loss_value)

            if warmup_lr_scheduler:
                warmup_lr_scheduler.step()
                if cur_iterations > warmup_iters+1:
                    print('removing warmup_lr_scheduler')
                    warmup_lr_scheduler = None

            if cur_iterations%summary_iters == 0:
                if model_name == 'yolov5':
                    model.ema.update_attr(model.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

                train_loss = np.mean(train_losses)

                eval_time = time.time()
                model.eval()
                pred = xview3.infer.inference_chip.run_eval(
                    model,
                    val_loader,
                    chips_path=chips_path,
                    device=device,
                    clip_boxes=clip_boxes,
                    bbox_size=bbox_size,
                    half=half_enabled,
                )
                model.train()

                if len(pred) > 0:
                    # Only keep top 4*len(gt) scoring points since otherwise NMS could be too slow.
                    pred = pred.nlargest(4*len(gt_incl_low), columns='score')
                    pred = xview3.eval.prune.nms(pred, distance_thresh=10)
                    pred = pred.reset_index(drop=True)
                    pred = xview3.eval.metric.drop_low_confidence_preds(pred, gt_incl_low, costly_dist=True)

                    # First test without near-shore. Then add near-shore on the threshold with highest loc_fscore.
                    val_scores = None
                    best_threshold = None
                    for threshold in [0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                        cur_pred = pred[pred.score >= threshold].reset_index(drop=True)
                        scores, _ = xview3.eval.metric.score(cur_pred, gt, shore_root=None, distance_tolerance=200, quiet=True, costly_dist=True)
                        if val_scores is None or scores['loc_fscore'] > val_scores['loc_fscore']:
                            val_scores = scores
                            best_threshold = threshold

                    if os.path.exists(shore_root):
                        cur_pred = pred[pred.score >= best_threshold].reset_index(drop=True)
                        val_scores, _ = xview3.eval.metric.score(cur_pred, gt, shore_root=shore_root, distance_tolerance=200, quiet=True, costly_dist=True)
                else:
                    print('got zero predictions, skipping evaluation')
                    val_scores = {'loc_fscore': 0.0, 'loc_fscore_shore': 0.0}

                val_score = val_scores['loc_fscore'] + val_scores['loc_fscore_shore']/5

                summary_writer.add_scalar('train_loss', train_loss, summary_epoch)
                for k, v in val_scores.items():
                    summary_writer.add_scalar(k, v, summary_epoch)

                print('summary_epoch {}: train_loss={} val={} elapsed={},{} lr={}'.format(
                    summary_epoch,
                    train_loss,
                    val_scores,
                    int(eval_time-summary_prev_time),
                    int(time.time()-eval_time),
                    optimizer.param_groups[0]['lr'],
                ))

                del train_losses[:]
                summary_epoch += 1
                summary_prev_time = time.time()

                # update the learning rate
                if warmup_lr_scheduler is None:
                    lr_scheduler.step(train_loss)

                # Model saving.
                if ema_factor:
                    state_dict = model.shadow.state_dict()
                else:
                    state_dict = model.state_dict()

                torch.save(state_dict, os.path.join(save_path, 'last.pth'))

                if val_score > best_score:
                    torch.save(state_dict, os.path.join(save_path, 'best.pth'))
                    best_score = val_score

                if summary_epoch%summary_save_freq == 0:
                    checkpoint_path = os.path.join(save_path, f"trained_model_{summary_epoch}_epochs.pth")
                    torch.save(state_dict, checkpoint_path)

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_path)
    main(config)

# sample usage: python src/xview3/training/train.py src/xview3/training/training_config.txt
