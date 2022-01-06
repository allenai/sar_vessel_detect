import argparse
import configparser
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import skimage.io
import skimage.transform
import sys
import time
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm

sys.path.insert(1, '/home/xview3/src') # use an appropriate path if not in the docker volume

from xview3.processing.constants import FISHING, NONFISHING, PIX_TO_M
from xview3.eval.prune import nms, confidence_pruning
from xview3.postprocess.v2.model_simple import Model
import xview3.models
from xview3.utils import clip
import xview3.transforms
import xview3.eval.ensemble


# Map from channel names to filenames.
channel_map = {
    'vv': 'VV_dB.tif',
    'vh': 'VH_dB.tif',
    'bathymetry': 'bathymetry.tif',
    'wind_speed': 'owiWindSpeed.tif',
    'wind_quality': 'owiWindQuality.tif',
    'wind_direction': 'owiWindDirection.tif',
}


def center(coord):
    return (coord[0] + (coord[2] / 2), coord[1] + (coord[3] / 2))


class SceneDataset(object):
    def __init__(self, image_folder, scene_ids, channels):
        self.image_folder = image_folder
        self.scene_ids = scene_ids
        self.channels = channels

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        scene_id = self.scene_ids[idx]

        # Load scene channels.
        # We always load bathymetry so we can eliminate detections on land.
        def get_channel(channel, shape=None):
            path = os.path.join(self.image_folder, scene_id, channel_map[channel])
            print(scene_id, 'read', path)
            cur = skimage.io.imread(path)
            cur = torch.tensor(cur, dtype=torch.float32)

            # If not same size as first channel, resample before chipping
            # to ensure chips from different channels are co-registered
            if shape is not None and cur.shape != shape:
                cur = torchvision.transforms.functional.resize(img=cur.unsqueeze(0), size=shape)[0, :, :]

            return cur

        im_channels = [get_channel(self.channels[0])] # nb this precludes vv/vh being first channel
        for channel in self.channels[1:]:
            if channel == "vv_over_vh":
                vvovervh = get_channel("vv", shape=im_channels[0].shape) / get_channel("vh", shape=im_channels[0].shape)
                vvovervh = np.nan_to_num(vvovervh, nan=0, posinf=0, neginf=0)
                im_channels.append(torch.tensor(vvovervh, dtype=torch.float32))
            else:
                im_channels.append(get_channel(channel, shape=im_channels[0].shape))

        # Stack channels.
        im = torch.stack(im_channels, dim=0)

        print(scene_id, 'done reading')
        return scene_id, im

def process_scene(args, clip_boxes, bbox_size, device, weight_files, model, postprocess_model, detector_transforms, postprocess_transforms, scene_id, im):
    with torch.no_grad():
        if im.shape[1] < args.window_size or im.shape[2] < args.window_size:
            raise Exception('image for scene {} is smaller than window size'.format(scene_id))

        # Outputs for each member of the ensemble/test-time-augmentation.
        member_outputs = []

        member_infos = [(0, 0, False, False)]
        #member_infos = [(0, 0, False, False), (0, 0, True, False), (757, 757, False, False), (1515, 1515, True, False)]
        #member_infos = [(0, 0, False, False), (757, 757, True, False)]
        #member_infos = [(0, 0, False, False), (0, 0, True, False), (0, 0, False, True), (0, 0, True, True)]

        for weight_file in weight_files:
            model.load_state_dict(torch.load(weight_file, map_location=device))
            for member_idx, (args_row_offset, args_col_offset, args_fliplr, args_flipud) in enumerate(member_infos):
                predicted_points = []

                # Loop over windows.
                row_offsets = [0] + list(range(
                    args.window_size-2*args.padding - args_row_offset,
                    im.shape[1]-args.window_size,
                    args.window_size-2*args.padding,
                )) + [im.shape[1]-args.window_size]
                col_offsets = [0] + list(range(
                    args.window_size-2*args.padding - args_col_offset,
                    im.shape[2]-args.window_size,
                    args.window_size-2*args.padding,
                )) + [im.shape[2]-args.window_size]

                member_start_time = time.time()
                for row_offset in row_offsets:
                    print('{} [{}/{}] (elapsed={})'.format(scene_id, row_offset, row_offsets[-1], time.time()-member_start_time))
                    for col_offset in col_offsets:
                        crop = im[:, row_offset:row_offset+args.window_size, col_offset:col_offset+args.window_size]
                        crop = torch.clone(crop)
                        crop, _ = detector_transforms(crop, None)
                        crop = crop[0:2, :, :]

                        if args_fliplr:
                            crop = torch.flip(crop, dims=[2])
                        if args_flipud:
                            crop = torch.flip(crop, dims=[1])

                        crop = crop.to(device)
                        output = model([crop])[0]
                        output = {k: v.to("cpu") for k, v in output.items()}

                        # Only keep output detections that are within bounds based
                        # on window size and padding.
                        keep_bounds = [
                            args.padding,
                            args.padding,
                            args.window_size - args.padding,
                            args.window_size - args.padding,
                        ]
                        if row_offset == 0:
                            keep_bounds[0] = 0
                        if col_offset == 0:
                            keep_bounds[1] = 0
                        if row_offset >= im.shape[1] - args.window_size:
                            keep_bounds[2] = args.window_size
                        if col_offset >= im.shape[2] - args.window_size:
                            keep_bounds[3] = args.window_size

                        keep_bounds[0] -= args.overlap
                        keep_bounds[1] -= args.overlap
                        keep_bounds[2] += args.overlap
                        keep_bounds[3] += args.overlap

                        for idx, box in enumerate(output["boxes"]):
                            # Determine the predicted point, in transformed image coordinates.
                            if clip_boxes:
                                # Boxes on edges of image might not be the right size.
                                if box[0] < bbox_size:
                                    pred_col = int(box[2] - bbox_size)
                                elif box[2] >= crop.shape[2]-bbox_size:
                                    pred_col = int(box[0] + bbox_size)
                                else:
                                    pred_col = int(np.mean([box[0], box[2]]))

                                if box[1] < bbox_size:
                                    pred_row = int(box[3] - bbox_size)
                                elif box[3] >= crop.shape[1]-bbox_size:
                                    pred_row = int(box[1] + bbox_size)
                                else:
                                    pred_row = int(np.mean([box[1], box[3]]))
                            else:
                                pred_row = int(np.mean([box[1], box[3]]))
                                pred_col = int(np.mean([box[0], box[2]]))

                            # Undo any transformations.
                            if args_fliplr:
                                pred_col = crop.shape[2] - pred_col
                            if args_flipud:
                                pred_row = crop.shape[1] - pred_row

                            # Compare against keep_bounds, which is pre-transformation.
                            if pred_row < keep_bounds[0] or pred_row >= keep_bounds[2]:
                                continue
                            if pred_col < keep_bounds[1] or pred_col >= keep_bounds[3]:
                                continue

                            label = output["labels"][idx].item()
                            is_fishing = label == FISHING
                            is_vessel = label in [FISHING, NONFISHING]
                            if "lengths" in output:
                                length = output["lengths"][idx].item()
                            else:
                                length = 0
                            score = output["scores"][idx].item()

                            scene_pred_row = row_offset + pred_row
                            scene_pred_col = col_offset + pred_col

                            predicted_points.append([
                                scene_pred_row,
                                scene_pred_col,
                                scene_id,
                                is_vessel,
                                is_fishing,
                                length,
                                score,
                            ])

                member_pred = pd.DataFrame(
                    data=predicted_points,
                    columns=(
                        "detect_scene_row",
                        "detect_scene_column",
                        "scene_id",
                        "is_vessel",
                        "is_fishing",
                        "vessel_length_m",
                        "score",
                    ),
                )
                print("[ensemble-member {}] {} detections found".format(member_idx, len(member_pred)))
                member_outputs.append(member_pred)

        # Merge ensemble members into one dataframe.
        pred = xview3.eval.ensemble.merge(member_outputs)

        # Pruning Code
        if args.nms_thresh is not None:
            pred = nms(pred, distance_thresh=args.nms_thresh)
        if args.conf is not None:
            pred = confidence_pruning(pred, threshold=args.conf)

        # Postprocessing Code
        bs = 32
        crop_size = 128
        pred = pred.reset_index(drop=True)
        for x in range(0, len(pred), bs):
            batch_df = pred.iloc[x : min((x+bs), len(pred))]

            crops, indices = [], []
            for idx,b in batch_df.iterrows():
                indices.append(idx)
                row, col = b['detect_scene_row'], b['detect_scene_column']

                crop = im[:, row-crop_size//2:row+crop_size//2, col-crop_size//2:col+crop_size//2]
                crop = torch.clone(crop)
                crop, _ = postprocess_transforms(crop, None)
                crop = crop[:, 8:120, 8:120]
                crop = torch.nn.functional.pad(crop, (8, 8, 8, 8))
                crops.append(crop)

            t = postprocess_model(torch.stack(crops, dim=0).to(device))
            t = [tt.cpu() for tt in t]
            pred_length, pred_confidence, pred_correct, pred_source, pred_fishing, pred_vessel = t

            for i in range(len(indices)):
                index = x + i

                # Prune confidence=LOW.
                if pred_confidence[i, :].argmax() == 0 and args.mode == 'full':
                    elim_inds.append(index)
                    continue

                pred.loc[index, 'vessel_length_m'] = pred_length[i].item()

                if args.mode in ['full', 'attribute']:
                    pred.loc[index, 'fishing_score'] = pred_fishing[i, 1].item()
                    pred.loc[index, 'vessel_score'] = pred_vessel[i, 1].item()
                    pred.loc[index, 'low_score'] = pred_confidence[i, 0].item()
                    pred.loc[index, 'is_fishing'] = (pred_fishing[i, 1] > 0.5).item() & (pred_vessel[i, 1] > 0.5).item()
                    pred.loc[index, 'is_vessel'] = (pred_vessel[i, 1] > 0.5).item()
                    pred.loc[index, 'correct_score'] = pred_correct[i, 1].item()

                if args.mode == 'full':
                    pred.loc[index, 'score'] = pred_correct[i, 1].item()

    if args.drop_cols:
        good_columns = [
            'detect_scene_row',
            'detect_scene_column',
            'scene_id',
            'is_vessel',
            'is_fishing',
            'vessel_length_m',
        ]
        bad_columns = []
        for column_name in pred.columns:
            if column_name in good_columns:
                continue
            bad_columns.append(column_name)
        pred = pred.drop(columns=bad_columns)

    if args.vessels_only:
        pred = pred[pred.is_vessel == True]

    if args.save_crops:
        pred = pred.reset_index(drop=True)
        detect_ids = [None]*len(pred)

        for index, label in pred.iterrows():
            row, col = label['detect_scene_row'], label['detect_scene_column']

            vh = im[0, row-crop_size//2:row+crop_size//2, col-crop_size//2:col+crop_size//2].numpy()
            vv = im[1, row-crop_size//2:row+crop_size//2, col-crop_size//2:col+crop_size//2].numpy()
            vh = np.clip((vh+50)*255/70, 0, 255).astype('uint8')
            vv = np.clip((vv+50)*255/70, 0, 255).astype('uint8')

            detect_id = '{}_{}'.format(scene_id, index)
            detect_ids[index] = detect_id
            skimage.io.imsave(os.path.join(args.output, '{}_vh.png'.format(detect_id)), vh)
            skimage.io.imsave(os.path.join(args.output, '{}_vv.png'.format(detect_id)), vv)

        pred.insert(len(pred.columns), 'detect_id', detect_ids)

    return pred

def main(args, config):
    # Create output directories if it does not already exist
    Path(os.path.split(args.output)[0]).mkdir(parents=True, exist_ok=True)

    # os.environ["CUDA_VISIBLE_DEVICES"]="3"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    channels = ['vh', 'vv', 'bathymetry']
    model_name = config.get("training", "Model")
    transform_names = config.get("data", "Transforms").split(",")
    clip_boxes = config.getboolean("data", "ClipBoxes", fallback=False)
    bbox_size = config.getint("data", "BboxSize", fallback=5)

    if args.scene_id and args.scene_id != 'all':
        scene_ids = args.scene_id.split(',')
    else:
        scene_ids = os.listdir(args.image_folder)

    transform_info = {
        'channels': channels,
        'bbox_size': bbox_size,
    }
    detector_transforms = xview3.transforms.get_transforms(transform_names, transform_info)
    postprocess_transforms = xview3.transforms.get_transforms(['CustomNormalize3'], transform_info)

    dataset = SceneDataset(
        image_folder=args.image_folder,
        scene_ids=scene_ids,
        channels=channels,
    )

    model_cls = xview3.models.models[model_name]
    model = model_cls(
        num_classes=4,
        num_channels=len(channels),
        image_size=args.window_size,
        device=device,
        config=config["training"],
        disable_multihead=True,
    )

    weight_files = args.weights.split(',')

    model.load_state_dict(torch.load(weight_files[0], map_location=device))
    model.to(device)
    model.eval()

    postprocess_model = Model()
    postprocess_model.load_state_dict(torch.load(args.postprocess_weights))
    postprocess_model.eval()
    postprocess_model.to(device)

    preds = []
    for scene_id, im in dataset:
        print('processing scene', scene_id)
        preds.append(process_scene(args, clip_boxes, bbox_size, device, weight_files, model, postprocess_model, detector_transforms, postprocess_transforms, scene_id, im))

    pred = pd.concat(preds)

    out_path = args.output
    if args.save_crops:
        out_path = os.path.join(args.output, 'predictions.csv')
    pred.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on xView3 reference model."
    )

    parser.add_argument("--image_folder", help="Path to the xView3 images")
    parser.add_argument("--scene_id", help="Test scene ID", default=None)
    parser.add_argument("--weights", help="Path to trained model weights")
    parser.add_argument("--output", help="Path in which to output inference CSVs")
    parser.add_argument("--config_path", help="Path to training configuration")
    parser.add_argument("--padding", type=int, help="Padding between sliding window", default=128)
    parser.add_argument("--window_size", type=int, help="Inference sliding window size", default=1024)
    parser.add_argument("--overlap", type=int, help="Overlap allowed for predictions between windows", default=0)

    # pruning
    parser.add_argument("--nms_thresh", type=int, help="Run NMS, with this threshold", default=None)
    parser.add_argument("--conf", type=float, help="Run confidence pruning, with this threshold", default=None)
    parser.add_argument("--drop_cols", help="Drop columns like score and bathymetry", default=False)

    # postprocessing
    parser.add_argument("--postprocess_weights", help="Path to the postprocessing model weights")
    parser.add_argument("--mode", type=str, help="Postprocessing mode, probably use attribute")

    # skylight
    parser.add_argument("--vessels_only", help="Only keep vessel predictions", default=False)
    parser.add_argument("--save_crops", help="Write crops around predicted points, in this case output is treated as a directory", default=False)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    main(args, config)
