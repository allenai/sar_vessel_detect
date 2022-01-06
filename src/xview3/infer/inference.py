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
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm

sys.path.insert(1, '/home/xview3/src') # use an appropriate path if not in the docker volume

from xview3.processing.constants import FISHING, NONFISHING, PIX_TO_M
import xview3.models
from xview3.utils import clip
import xview3.transforms


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
    def __init__(self, image_folder, scene_ids, channels, transforms):
        self.image_folder = image_folder
        self.scene_ids = scene_ids
        self.channels = channels
        self.transforms = transforms

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

        # Stack channels and apply transforms.
        im = torch.stack(im_channels, dim=0)
        im, _ = self.transforms(im, None)

        print(scene_id, 'done reading')
        return scene_id, im

def main(args, config):
    if args.scene_ids is not None:
        scene_ids = args.scene_ids.split(",")
    elif args.scene_path is not None:
        with open(args.scene_path, 'r') as f:
            scene_ids = [line.strip() for line in f.readlines() if line.strip()]
    else:
        scene_ids = os.listdir(args.image_folder)

    # Create output directories if it does not already exist
    Path(os.path.split(args.output)[0]).mkdir(parents=True, exist_ok=True)

    # os.environ["CUDA_VISIBLE_DEVICES"]="3"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    channels = config.get("data", "Channels").strip().split(",")
    model_name = config.get("training", "Model")
    transform_names = config.get("data", "Transforms").split(",")
    clip_boxes = config.getboolean("data", "ClipBoxes", fallback=False)
    bbox_size = config.getint("data", "BboxSize", fallback=5)

    transforms = xview3.transforms.get_transforms(transform_names, {
        'channels': channels,
        'bbox_size': bbox_size,
    })
    dataset = SceneDataset(
        image_folder=args.image_folder,
        scene_ids=scene_ids,
        channels=channels,
        transforms=transforms,
    )

    model_cls = xview3.models.models[model_name]
    model = model_cls(
        num_classes=4,
        num_channels=len(channels),
        image_size=args.window_size,
        device=device,
        config=config["training"],
    )

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    df_out = []

    with torch.no_grad():
        for scene_id, im in tqdm(dataset):
            if im.shape[1] < args.window_size or im.shape[2] < args.window_size:
                raise Exception('image for scene {} is smaller than window size'.format(scene_id))

            # Loop over windows.
            row_offsets = [0] + list(range(
                args.window_size-2*args.padding - args.row_offset,
                im.shape[1]-args.window_size,
                args.window_size-2*args.padding,
            )) + [im.shape[1]-args.window_size]
            col_offsets = [0] + list(range(
                args.window_size-2*args.padding - args.col_offset,
                im.shape[2]-args.window_size,
                args.window_size-2*args.padding,
            )) + [im.shape[2]-args.window_size]

            for row_offset in row_offsets:
                print(scene_id, row_offset, '/', row_offsets[-1])
                for col_offset in col_offsets:
                    crop = im[:, row_offset:row_offset+args.window_size, col_offset:col_offset+args.window_size]

                    if args.fliplr:
                        crop = torch.flip(crop, dims=[2])
                    if args.flipud:
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
                        if args.fliplr:
                            pred_col = crop.shape[2] - pred_col
                        if args.flipud:
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

                        df_out.append([
                            scene_pred_row,
                            scene_pred_col,
                            scene_id,
                            is_vessel,
                            is_fishing,
                            length,
                            score,
                        ])

    df_out = pd.DataFrame(
        data=df_out,
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
    df_out.to_csv(args.output, index=False)
    print(f"{len(df_out)} detections found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on xView3 reference model."
    )

    parser.add_argument("--image_folder", help="Path to the xView3 images")
    parser.add_argument(
        "--scene_ids", help="Comma separated list of test scene IDs", default=None
    )
    parser.add_argument("--scene_path", help="Path to scene list txt", default=None)
    parser.add_argument("--weights", help="Path to trained model weights")
    parser.add_argument("--output", help="Path in which to output inference CSVs")
    parser.add_argument("--config_path", help="Path to training configuration")
    parser.add_argument("--batch_size", type=int, help="Inference batch size", default=1)
    parser.add_argument("--padding", type=int, help="Padding between sliding window", default=128)
    parser.add_argument("--window_size", type=int, help="Inference sliding window size", default=1024)
    parser.add_argument("--overlap", type=int, help="Overlap allowed for predictions between windows", default=0)

    # augmentations
    parser.add_argument("--fliplr", type=bool, help="Left-right flip (augmentation)", default=False)
    parser.add_argument("--flipud", type=bool, help="Vertical flip (augmentation)", default=False)
    parser.add_argument("--row_offset", type=int, help="Row offset (augmentation)", default=0)
    parser.add_argument("--col_offset", type=int, help="Column offset (augmentation)", default=0)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    main(args, config)
