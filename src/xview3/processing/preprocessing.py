import glob
import json
import os
import time
import configparser
from pathlib import Path
import sys

sys.path.insert(1, '/home/xview3/src') # use an appropriate path if not in the docker volume

import numpy as np
import pandas as pd
import math
import rasterio
from rasterio.enums import Resampling

from xview3.processing.constants import BACKGROUND, FISHING, NONFISHING, NONVESSEL

def pad(vh, rows, cols,overlap):
    """
    Pad an image to make it divisible by some block_size.
    Pad on the right and bottom edges so annotations are still usable.
    """
    r, c = vh.shape
    to_rows = math.ceil(r / rows) * rows + overlap * 2
    to_cols = math.ceil(c / cols) * cols + overlap * 2
    pad_rows = to_rows - r - overlap
    pad_cols = to_cols - c - overlap
    vh_pad = np.pad(
        vh, pad_width=((overlap, pad_rows), (overlap, pad_cols)), mode="constant", constant_values=0
    )
    return vh_pad, pad_rows, pad_cols


def chip_sar_img(input_img, sz, overlap):
    """
    Takes a raster from xView3 as input and outputs
    a set of chips and the coordinate grid for a
    given chip size

    Args:
        input_img (numpy.array): Input image in np.array form
        sz (int): Size of chip (will be sz x sz x # of channlls)

    Returns:
        images: set of image chips
        images_grid: grid coordinates for each chip
    """
    # The input_img is presumed to already be padded
    images,nrows,ncols = view_as_blocks(input_img, (sz, sz), overlap)
    images_grid = (nrows,ncols)
    
    return images, images_grid


def view_as_blocks(arr, block_size, overlap):
    """
    Break up an image into blocks and return array.
    """
    m, n = arr.shape
    # assume square
    chip_size,_ = block_size
    stride = chip_size
    step = chip_size + 2 * overlap

    nrows, ncols = (m - 2*overlap) // chip_size, (n - 2*overlap) // chip_size
    splitted = []
    for i in range(nrows):
        for j in range(ncols):
            h_start = j*stride
            v_start = i*stride
            cropped = arr[v_start:v_start+step, h_start:h_start+step]
            splitted.append(cropped)
    return splitted,nrows,ncols


def find_nearest(lon, lat, x, y):
    """Find nearest row/col pixel for x/y coordinate.
    lon, lat : 2D image
    x, y : scalar coordinates
    """
    X = np.abs(lon - x)
    Y = np.abs(lat - y)
    return np.where((X == X.min()) & (Y == Y.min()))


def get_grid_coords(padded_img, chips, grids, overlap):
    """
    Obtain grid coordinates for chip origins from padded images
    """
    chip_size = chips[0].shape[0] - 2*overlap 
    grid_coords_y = np.linspace(0, padded_img.shape[0] - chip_size - overlap*2, grids[0])
    grid_coords_x = np.linspace(0, padded_img.shape[1] - chip_size - overlap*2, grids[1])
    grid_coords = [(int(x), int(y)) for y in grid_coords_y for x in grid_coords_x]
    return grid_coords


def process_scene(
    scene_id,
    detections,
    channels,
    chip_size,
    overlap_width,
    chips_path,
    overwrite_preproc,
    root,
    index,
):
    """
    Preprocess scene by loading images, chipping them,
    saving chips and grid coordinates, converting scene-level to
    chip-level detections, and returning a dataframe with
    information required for training.
    """

    pixel_detections = pd.DataFrame()
    no_image_data_count = 0

    # If detections file exists, load chip-level annotations for the scene;
    # otherwise, assume this is for inference

    if detections is not None:
        scene_detects = detections[detections["scene_id"] == scene_id]
        scene_detect_num = len(scene_detects)
        print(
            f"Detections expected for scene # {index} ({scene_id}): {scene_detect_num}"
        )
        # Loading up detections file
        if not overwrite_preproc and os.path.exists(
            f"{chips_path}/chip_annotations.csv"
        ):
            pixel_detections = pd.read_csv(f"{chips_path}/chip_annotations.csv")
            pixel_detections = pixel_detections[
                (pixel_detections["scene_id"] == scene_id)
            ]
    else:
        scene_detect_num = 0
        print(f"No detection file, only chipping for inference")


    # Use xView3 data file structure to define files to load up
    # for each possible channel
    files = {}
    files["vh"] = os.path.join(root, f"{scene_id}", "VH_dB.tif")
    files["vv"] = Path(files["vh"]).parent / "VV_dB.tif"
    files["bathymetry"] = Path(files["vh"]).parent / "bathymetry.tif"
    files["wind_speed"] = Path(files["vh"]).parent / "owiWindSpeed.tif"
    files["wind_direction"] = Path(files["vh"]).parent / "owiWindDirection.tif"
    files["wind_quality"] = Path(files["vh"]).parent / "owiWindQuality.tif"
    files["mask"] = Path(files["vh"]).parent / "owiMask.tif"

    imgs, chips, grids = {}, {}, {}

    # For each channel, if it is already chipped, do not re-chip
    for fl in channels:
        temp_folder = Path(chips_path) / scene_id / fl
        if os.path.exists(temp_folder) and (not overwrite_preproc):
            print(f"Using existing preprocessed {fl} data for scene {scene_id}")
            continue
        else:
            os.makedirs(temp_folder, exist_ok=True)
        src = rasterio.open(files[fl])
        imgs[fl] = src.read(1)

        # If not same size as first channel, resample before chipping
        # to ensure chips from different channels are co-registered
        if not imgs[fl].shape == imgs[channels[0]].shape:
            imgs[fl] = src.read(
                out_shape=(
                    imgs[channels[0]].shape[0],
                    imgs[channels[0]].shape[1],
                ),
                resampling=Resampling.bilinear,
            ).squeeze()
        try:
            assert imgs[fl].shape == imgs[channels[0]].shape
        except AssertionError as e:
            print(f"imgs[fl].shape = {imgs[fl].shape}")
            print(f"imgs[channels[0]].shape = {imgs[channels[0]].shape}")
            raise AssertionError()

        # Pad the raster to be a multiple of the chip size
        padded_img, _, _ = pad(imgs[fl], chip_size, chip_size, overlap_width)

        # Get image chips and grids
        chips[fl], grids[fl] = chip_sar_img(padded_img, chip_size, overlap_width)

        # Saving chips
        for i in range(len(chips[fl])):
            chip = chips[fl][i]
            if np.max(np.max(chip)) == -32768:
                no_image_data_count += 1
                continue
            np.save(f"{temp_folder}/{i}_{fl}.npy", chip.astype(np.float16))

        if fl == channels[0]:
            # Getting grid coordinates
            grid_coords = get_grid_coords(padded_img, chips[fl], grids[fl], overlap_width)

            # Saving offsets for each chip; these offsets are alsp needed to convert
            # chip-level predictions to scene-level predictions at
            # inference time
            write_object = {
                "offsets": grid_coords,
            }
            json.dump(
                write_object, open(Path(chips_path) / scene_id / "coords.json", "w")
            )

            if detections is not None:
                print("Getting detections...")
                # Get pixel values for detections in scene
                (scene_detects.loc[:,"scene_rows"], scene_detects.loc[:,"scene_cols"],) = (
                    scene_detects["detect_scene_row"]+overlap_width,
                    scene_detects["detect_scene_column"]+overlap_width,
                )

                # Convert scene-level detection coordinates to chip-level annotations

                for chip in enumerate(grid_coords):
                    c1,c2,r1,r2 = chip[1][0],chip[1][0]+chip_size+overlap_width*2, chip[1][1], chip[1][1]+chip_size+overlap_width*2
                    this_chip_detections = scene_detects[(scene_detects["scene_rows"]>= r1) & (scene_detects["scene_rows"]<= r2) & 
                                                         (scene_detects["scene_cols"]>= c1) & (scene_detects["scene_cols"]<= c2)]
                    for index, det in this_chip_detections.iterrows():
                        det["rows"] = det["scene_rows"] - r1
                        det["columns"] = det["scene_cols"] - c1
                        det["chip_index"] = chip[0]
                        pixel_detections = pixel_detections.append(det,ignore_index=True)
                    
                intcols = ['vessel_class','scene_rows','scene_cols','rows','columns','chip_index','detect_scene_row','detect_scene_column']

                pixel_detections['is_vessel'] = pixel_detections['is_vessel'].replace({0:False, 1:True, np.nan:np.nan})
                pixel_detections['is_fishing'] = pixel_detections['is_fishing'].replace({0:False, 1:True, np.nan:np.nan})
                pixel_detections.loc[:, intcols] = pixel_detections.loc[:, intcols].astype(int)

                if not os.path.exists(f"{chips_path}/chip_annotations.csv"):
                    pixel_detections.to_csv(
                        f"{chips_path}/chip_annotations.csv",
                        mode="w",
                        header=True,
                    )
                else:
                    pixel_detections.to_csv(
                        f"{chips_path}/chip_annotations.csv",
                        mode="a",
                        header=False,
                    )

    # Print number of detections per scene; make sure it aligns with
    # number expected
    if detections is not None:
        if not overwrite_preproc:
            chip_detect_num = len(
                pixel_detections[
                    (pixel_detections["scene_id"] == scene_id)
                    & (pixel_detections["vessel_class"] != BACKGROUND)
                ]
            )
        else:
            chip_detect_num = len(pixel_detections)

        print(
            f"Detections recovered in chips for scene {scene_id}: {chip_detect_num} "
        )
    
    print(f"{no_image_data_count} chips across {len(channels)} image channels had no image data and were not saved \n")

    return pixel_detections

def main(config):
    image_folder = config.get("locations", "ImageFolder")
    label_file = config.get("locations", "LabelFile",fallback=None)
    chips_path = config.get("locations", "ChipsPath")
    use_scene_list = config.getboolean("locations", "UseSceneList")
    config_scene_list = config.get("locations", "SceneList").strip().split(",")

    num_preproc_workers = config.getint("chip_params", "NumPreprocWorkers")
    overwrite_preproc = config.getboolean("chip_params", "OverwritePreprocessing")
    is_distributed = config.getboolean("chip_params", "IsDistributed")
    channels = config.get("chip_params", "Channels").strip().split(",")
    chip_size = config.getint("chip_params", "ChipSize")
    overlap_width = config.getint("chip_params", "OverlapWidth")

    if not use_scene_list:
        scenes = [
            a.strip("\n").strip("/").split("/")[-1][:67] for a in os.listdir(image_folder)
        ]
    else:
        scenes = config_scene_list


    ##### TODO: dig into here, does this positively assign detections without a vessel_class to nonvessel?
    if label_file:
        detections = pd.read_csv(label_file, low_memory=False)
        vessel_class = []
        for ii, row in detections.iterrows():
            if row.is_vessel and row.is_fishing:
                vessel_class.append(FISHING)
            elif row.is_vessel and not row.is_fishing:
                vessel_class.append(NONFISHING)
            elif not row.is_vessel:
                vessel_class.append(NONVESSEL)
        detections["vessel_class"] = vessel_class
        # # Assuming we're only using examples with vessel class for this
        # # training procedure
        # if self.ais_only:
        #     detections = detections.dropna(subset=["vessel_class"])
    else:
        detections = None
    
    # Logic for overwriting existing detectinos file
    if overwrite_preproc and os.path.exists(
        f"{chips_path}/chip_annotations.csv"
    ):
        os.remove(f"{chips_path}/chip_annotations.csv")


    start = time.time()
    for jj, scene_id in enumerate(scenes):
        print(f"Processing scene {jj} of {len(scenes)}...")
        tmp=process_scene(
                scene_id,
                detections,
                channels,
                chip_size,
                overlap_width,
                chips_path,
                overwrite_preproc,
                image_folder,
                jj,
            )

    el = time.time() - start
    print(f"Elapsed Time: {np.round(el/60, 2)} Minutes")



if __name__ == "__main__":

    config_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_path)
    main(config)

    # sample usage: python src/xview3/processing/preprocessing.py src/xview3/processing/chipping_config.txt
