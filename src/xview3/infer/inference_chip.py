import argparse
import configparser
import json
import numpy as np
import os
import os.path
import pandas as pd
from pathlib import Path
import sys
import torch
import torch.cuda.amp
import torchvision
from tqdm import tqdm

sys.path.insert(1, '/home/xview3/src') # use an appropriate path if not in the docker volume

from xview3.processing.constants import FISHING, NONFISHING, PIX_TO_M
from xview3.processing.dataloader import SARDataset
import xview3.models
import xview3.transforms
import xview3.training.utils

def run_eval(model, loader, device, chips_path, clip_boxes=False, bbox_size=5, half=False):
    # Map from scene_id to list of chip offsets.
    # And provide helper function to obtain this data for (scene_id, chip_idx).
    chip_offsets = {}
    def get_chip_offset(scene_id, chip_idx):
        if scene_id not in chip_offsets:
            with open(os.path.join(chips_path, scene_id, 'coords.json'), 'r') as f:
                chip_offsets[scene_id] = json.load(f)['offsets']

        return chip_offsets[scene_id][chip_idx]

    df_out = []

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = list(image.to(device) for image in images)

            with torch.cuda.amp.autocast(enabled=half):
                outputs = model(images)

            outputs = [{k: v.to("cpu") for k, v in output.items()} for output in outputs]

            for img_idx, output in enumerate(outputs):
                image = images[img_idx]
                target = targets[img_idx]
                scene_id = target['scene_id']
                chip_idx = target['chip_id']
                col_offset, row_offset = get_chip_offset(scene_id, chip_idx)

                for box_idx, box in enumerate(output["boxes"]):
                    if clip_boxes:
                        # Boxes on edges of image might not be the right size.
                        if box[0] < bbox_size:
                            pred_col = int(box[2] - bbox_size)
                        elif box[2] >= image.shape[2]-bbox_size:
                            pred_col = int(box[0] + bbox_size)
                        else:
                            pred_col = int(np.mean([box[0], box[2]]))

                        if box[1] < bbox_size:
                            pred_row = int(box[3] - bbox_size)
                        elif box[3] >= image.shape[1]-bbox_size:
                            pred_row = int(box[1] + bbox_size)
                        else:
                            pred_row = int(np.mean([box[1], box[3]]))
                    else:
                        pred_row = int(np.mean([box[1], box[3]]))
                        pred_col = int(np.mean([box[0], box[2]]))

                    label = output["labels"][box_idx].item()
                    is_fishing = label == FISHING
                    is_vessel = label in [FISHING, NONFISHING]
                    if "lengths" in output:
                        length = output["lengths"][box_idx].item()
                    else:
                        length = 0
                    if "fishing_scores" in output:
                        fishing_score = output["fishing_scores"][box_idx].item()
                    else:
                        fishing_score = 0
                    if "vessel_scores" in output:
                        vessel_score = output["vessel_scores"][box_idx].item()
                    else:
                        vessel_score = 0
                    score = output["scores"][box_idx].item()

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
                        fishing_score,
                        vessel_score,
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
            "fishing_score",
            "vessel_score",
        ),
    )
    return df_out


def main(args, config):
    # Create output directories if it does not already exist
    Path(os.path.split(args.output)[0]).mkdir(parents=True, exist_ok=True)

    # os.environ['CUDA_VISIBLE_DEVICES']="1"
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

    scene_list = None
    if args.scene_ids is not None:
        scene_list = args.scene_ids.split(',')

    dataset = SARDataset(
        chips_path=args.chips_path,
        scene_path=args.scene_path,
        scene_list=scene_list,
        transforms=transforms,
        channels=channels,
        all_chips=True,
        geosplit=args.geosplit,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_loader_workers,
        collate_fn=xview3.training.utils.collate_fn,
    )

    model_cls = xview3.models.models[model_name]
    image_size = dataset[0][0].shape[1]
    print('image_size={}'.format(image_size))
    model = model_cls(
        num_classes=4,
        num_channels=len(channels),
        device=device,
        config=config["training"],
        image_size=image_size,
    )

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    df_out = run_eval(
        model,
        loader,
        chips_path=args.chips_path,
        device=device,
        clip_boxes=clip_boxes,
        bbox_size=bbox_size,
    )
    df_out.to_csv(args.output, index=False)
    print(f"{len(df_out)} detections found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on xView3 reference model."
    )

    parser.add_argument("--chips_path", help="Path to the xView3 chips")
    parser.add_argument(
        "--scene_ids", help="Comma separated list of test scene IDs", default=None
    )
    parser.add_argument("--scene_path", help="Path to the scene split list", default=None)
    parser.add_argument("--weights", help="Path to trained model weights")
    parser.add_argument("--output", help="Path in which to output inference CSVs")
    parser.add_argument("--config_path", help="Path to training configuration")
    parser.add_argument("--batch_size", type=int, help="Inference batch size", default=8)
    parser.add_argument("--num_loader_workers", type=int, help="Number loader workers for inference", default=4)
    parser.add_argument("--geosplit", help="Geo-split even or odd", default=None)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    main(args, config)
