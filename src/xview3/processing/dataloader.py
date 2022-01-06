import glob
import json
import os

import math
import numpy as np
import pandas as pd
import random
import torch
from rasterio.enums import Resampling

from xview3.processing.constants import BACKGROUND, FISHING, NONFISHING, NONVESSEL
import xview3.utils

PRECHIPPED_CHANNELS = ["vh","vv","bathymetry","wind_speed","wind_direction","wind_quality","mask","vh_other","google"]

def get_valid_chips(channels, scene_path):
    scene_disk_chips = None
    for fl in channels:
        if fl not in PRECHIPPED_CHANNELS:
            continue
        fl_path = os.path.join(scene_path, fl)
        fl_chips = set([
            int(fname.split('_')[0])
            for fname in os.listdir(fl_path)
            if fname.endswith('.npy')
        ])
        if scene_disk_chips is None:
            scene_disk_chips = fl_chips
        else:
            scene_disk_chips = scene_disk_chips.intersection(fl_chips)
    return scene_disk_chips


def is_near_shore(info):
    chips_path, scene_id, chip_index = info
    bathymetry = np.load(os.path.join(chips_path, scene_id, 'bathymetry', '{}_bathymetry.npy'.format(chip_index)))
    if np.count_nonzero(bathymetry < 0) < 200*800:
        return False
    if np.count_nonzero(bathymetry > 0) < 200*800:
        return False
    return True


def get_latlon_channel(chips_path, scene_id, chip_index, dim, fl):
    latlon_dict = get_chip_corners(chips_path, scene_id)
    left_lon, top_lat, right_lon, bottom_lat = latlon_dict[chip_index]

    # Interpolate a channel based on the two corners and the dimensions
    if fl == 'lat':
        lats = np.linspace(top_lat, bottom_lat, dim)
        lats = np.expand_dims(lats, 0)
        lats = np.tile(lats, (dim, 1))
        lats = (lats - np.min(lats))/np.ptp(lats)  # normalize between [0,1]
        return lats
    if fl == 'lon':
        lons = np.linspace(left_lon, right_lon, dim)
        lons = np.expand_dims(lons, 1)
        lons = np.tile(lons, (1, dim))
        lons = (lons - np.min(lons))/np.ptp(lons)  # normalize between [0,1]
        return lons


histogram_cache = {}
def get_histogram_channel(chips_path, scene_id, chip_row, chip_col):
    if scene_id not in histogram_cache:
        histogram_cache[scene_id] = pd.read_csv(os.path.join(chips_path, scene_id, 'histogram.csv'))

    df = histogram_cache[scene_id]
    df = df[
        (df.detect_scene_row >= chip_row)
        & (df.detect_scene_row < chip_row+800)
        & (df.detect_scene_column >= chip_col)
        & (df.detect_scene_column < chip_col+800)
    ]
    histogram = np.zeros((800, 800), dtype='float32')
    for _, label in df.iterrows():
        row = label.detect_scene_row - chip_row
        col = label.detect_scene_column - chip_col
        histogram[row, col] += 1

    return histogram/100

overlap_cache = {}
def get_overlap_channel(chips_path, scene_id, chip_index):
    if scene_id not in overlap_cache:
        with open(os.path.join(chips_path, scene_id, 'histogram.json'), 'r') as f:
            overlap_cache[scene_id] = json.load(f)

    count = overlap_cache[scene_id][chip_index]
    im = count*np.ones((800, 800), dtype='float32')
    return im/500

def get_histogram2_channel(chips_path, scene_id, chip_index, chip_row, chip_col):
    histogram = 100*get_histogram_channel(chips_path, scene_id, chip_row, chip_col)
    overlap = 500*get_overlap_channel(chips_path, scene_id, chip_index)
    if overlap.min() == 0:
        return overlap
    else:
        return histogram/overlap

chip_corners_cache = {}
def get_chip_corners(chips_path, scene_id):
    if scene_id not in chip_corners_cache:
        with open(os.path.join(chips_path, scene_id, 'corners.json'), 'r') as f:
            chip_corners_cache[scene_id] = json.load(f)
    return chip_corners_cache[scene_id]

region_id_cache = {}
def get_region_id(chips_path, scene_id):
    if len(region_id_cache) == 0:
        for i in range(0, 5):
            with open('/xview3/all/splits/cluster/{}-all.txt'.format(i), 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    region_id_cache[line] = i
    return region_id_cache[scene_id]/4

class SARDataset(object):
    """
    Pytorch dataset for working with Sentinel-1 data
    """

    def __init__(
        self,
        transforms,
        scene_path=None,
        scene_list=None,
        chips_path=".",
        channels=["vh", "vv", "wind_direction"],
        bbox_size=5,
        clip_boxes=False,
        background_frac=None,
        background_min=3,
         # Only use medium/high confidence detections.
        skip_low_confidence=False,
        class_map=None,
        # Use box rather than point labels, and only use detections with box labels.
        use_box_labels=False,
        # Iterate over all chips rather than only chips with detections.
        # background_frac is ignored with this option.
        all_chips=False,
        # Only select chips that are near-shore (currently based on bathymetry).
        near_shore_only=False,
        # num_workers=1,
        span=1,
        custom_annotation_path=None,
        geosplit=None,
        histogram_hide_prob=None,
        chip_list=None,
        i2=False,
    ):

        self.bbox_size = bbox_size
        self.clip_boxes = clip_boxes
        self.background_frac = background_frac
        self.background_min = background_min
        self.chips_path = chips_path
        self.transforms = transforms
        self.channels = channels
        self.skip_low_confidence = skip_low_confidence
        self.class_map = class_map
        self.use_box_labels = use_box_labels
        self.span = span
        self.histogram_hide_prob = histogram_hide_prob
        # self.num_workers = num_workers
        self.coords = {}
        self.i2 = i2
        self.i2_option_cache = {}

        if custom_annotation_path:
            self.annotation_path = custom_annotation_path
        else:
            self.annotation_path = os.path.join(self.chips_path, 'chip_annotations.csv')

        # Getting image lst
        if scene_list:
            self.scenes = scene_list
        elif scene_path:
            with open(scene_path, 'r') as f:
                self.scenes = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.scenes = [
                scene_id for scene_id in os.listdir(chips_path)
                if os.path.isdir(os.path.join(chips_path, scene_id))
            ]

        # Get chip-level detection coordinates - should be available from preprocessing step
        self.pixel_detections=None
        self.pixel_detections = self.chip_and_get_pixel_detections()

        if self.skip_low_confidence:
            print("Removing low-confidence detections")
            orig_count = len(self.pixel_detections)
            self.pixel_detections = self.pixel_detections[
                self.pixel_detections.confidence.isin(['MEDIUM', 'HIGH'])
            ]
            print("... {} -> {}".format(orig_count, len(self.pixel_detections)))

        if self.use_box_labels:
            print("Removing detections with no box label")
            orig_count = len(self.pixel_detections)
            self.pixel_detections = self.pixel_detections[self.pixel_detections.left >= 0]
            print("... {} -> {}".format(orig_count, len(self.pixel_detections)))

        if all_chips or self.pixel_detections is None:
            self.chip_indices = []
            for scene_id in self.scenes:
                chips_with_data = get_valid_chips(self.channels, os.path.join(self.chips_path, scene_id))
                self.chip_indices += [(scene_id, a) for a in chips_with_data]
        else:
            # Add background chips for negative sampling
            if self.background_frac and (self.pixel_detections is not None):
                print("Adding background chips...")
                self.add_background_chips()

            self.chip_indices = list(
                set(
                    zip(
                        self.pixel_detections.scene_id, self.pixel_detections.chip_index
                    )
                )
            )

        if chip_list:
            with open(chip_list, 'r') as f:
                chip_list_ = json.load(f)
            chip_list_ = set([tuple(t) for t in chip_list_])
            new_chip_indices = [t for t in self.chip_indices if t in chip_list_]
            print('chip_list: {} -> {}'.format(len(self.chip_indices), len(new_chip_indices)))
            self.chip_indices = new_chip_indices

        if near_shore_only:
            # Select subset of chip_indices that are near-shore.
            print("Selecting near-shore chips...")
            import multiprocessing
            from tqdm import tqdm

            p = multiprocessing.Pool(8)
            inputs = [(self.chips_path, scene_id, chip_index) for scene_id, chip_index in self.chip_indices]
            near_shore_outputs = list(tqdm(p.imap(is_near_shore, inputs), total=len(inputs)))
            p.close()

            new_chip_indices = [self.chip_indices[i] for i, okay in enumerate(near_shore_outputs) if okay]
            print('near-shore: prune {} -> {}'.format(len(self.chip_indices), len(new_chip_indices)))
            self.chip_indices = new_chip_indices

        if geosplit:
            def geosplit_filter(t):
                scene_id, chip_index = t
                corners = get_chip_corners(self.chips_path, scene_id)[chip_index]
                left = int(math.floor(corners[0]))
                top = int(math.floor(corners[1]))
                right = int(math.floor(corners[2]))
                bottom = int(math.floor(corners[3]))

                if left != right or top != bottom:
                    # Exclude border tiles for even/odd training.
                    # But include for inference when specified split is "notodd".
                    if geosplit == 'notodd':
                        return True
                    else:
                        return False

                is_even = (left+top)%2 == 0
                return (geosplit in ['even', 'notodd'] and is_even) or (geosplit == 'odd' and not is_even)

            new_chip_indices = list(filter(geosplit_filter, self.chip_indices))
            print('geosplit: {} -> {}'.format(len(self.chip_indices), len(new_chip_indices)))
            self.chip_indices = new_chip_indices

        print('loading chip offsets')
        self.chip_offsets = {}
        for scene_id, _ in self.chip_indices:
            if scene_id in self.chip_offsets:
                continue
            with open(os.path.join(self.chips_path, scene_id, 'coords.json'), 'r') as f:
                self.chip_offsets[scene_id] = [(col, row) for col, row in json.load(f)['offsets']]

        print(f"Number of Unique Chips: {len(self.chip_indices)}")
        print("Initialization complete")

    def __len__(self):
        return len(self.chip_indices)

    def __getitem__(self, idx):
        # Load and condition image chip data
        scene_id, chip_index = self.chip_indices[idx]
        chip_col, chip_row = self.chip_offsets[scene_id][chip_index]

        if self.span == 2:
            if random.random() < 0.5:
                chip_row -= 800
            if random.random() < 0.5:
                chip_col -= 800

        histogram_hide = self.histogram_hide_prob is not None and random.random() < self.histogram_hide_prob

        data = np.ones((len(self.channels), 800*self.span, 800*self.span), dtype=np.float32)
        for channel_idx, fl in enumerate(self.channels):
            if fl in ['histogram', 'overlap', 'histogram2', 'lon', 'lat']:
                data[channel_idx, :, :] = -0.5
            if fl == 'regionid':
                data[channel_idx, :, :] = get_region_id(self.chips_path, scene_id)
            else:
                data[channel_idx, :, :] = -32768

        for off_row in range(0, 800*self.span, 800):
            for off_col in range(0, 800*self.span, 800):
                # Determine the chip index for this chip.
                # Skip chips that are outside the image bounds (leave as -32768).
                chip_k = (chip_col+off_col, chip_row+off_row)
                if chip_k not in self.chip_offsets[scene_id]:
                    continue

                cur_chip_index = self.chip_offsets[scene_id].index(chip_k)

                for channel_idx, fl in enumerate(self.channels):
                    if fl in PRECHIPPED_CHANNELS:
                        pth = f"{self.chips_path}/{scene_id}/{fl}/{int(cur_chip_index)}_{fl}.npy"
                        if not os.path.exists(pth):
                            continue
                        data[channel_idx, off_row:off_row+800, off_col:off_col+800] = np.load(pth)
                    elif fl == "vv_over_vh":
                        vv_pth = f"{self.chips_path}/{scene_id}/vv/{int(cur_chip_index)}_vv.npy"
                        vh_pth = f"{self.chips_path}/{scene_id}/vh/{int(cur_chip_index)}_vh.npy"
                        vvovervh = np.load(vv_pth) / np.load(vh_pth)
                        data[channel_idx, off_row:off_row+800, off_col:off_col+800] = np.nan_to_num(vvovervh, nan=0, posinf=0, neginf=0)
                    elif fl == 'lat' or fl == 'lon':
                        data[channel_idx, off_row:off_row+800, off_col:off_col+800] = get_latlon_channel(self.chips_path, scene_id, cur_chip_index, 800, fl)
                    elif fl in ['histogram', 'overlap', 'histogram2']:
                        if histogram_hide:
                            continue

                        if fl == 'histogram':
                            data[channel_idx, off_row:off_row+800, off_col:off_col+800] = get_histogram_channel(self.chips_path, scene_id, chip_row+off_row, chip_col+off_col)
                        elif fl == 'overlap':
                            data[channel_idx, off_row:off_row+800, off_col:off_col+800] = get_overlap_channel(self.chips_path, scene_id, cur_chip_index)
                        elif fl == 'histogram2':
                            data[channel_idx, off_row:off_row+800, off_col:off_col+800] = get_histogram2_channel(self.chips_path, scene_id, cur_chip_index, chip_row+off_row, chip_col+off_col)
                    elif fl == 'regionid':
                        # already set above
                        pass
                    else:
                        print(f"Unknown channel {fl}, cannot parse")

        if self.i2:
            i2_im = self.get_i2(scene_id, chip_index)
            data = np.concatenate([data, i2_im], axis=0)

        # Stacking channels to create multi-band image chip
        img = torch.as_tensor(data)

        # Get label information if it exists
        if self.pixel_detections is not None:
            centers = []
            boxes = []
            class_labels = []
            length_labels = []
            confidence_labels = []
            fishing_labels = []
            vessel_labels = []
            score_labels = []

            for off_row in range(0, 800*self.span, 800):
                for off_col in range(0, 800*self.span, 800):
                    chip_k = (chip_col+off_col, chip_row+off_row)
                    if chip_k not in self.chip_offsets[scene_id]:
                        continue

                    cur_chip_index = self.chip_offsets[scene_id].index(chip_k)

                    detects = self.pixel_detections[
                        (self.pixel_detections["scene_id"] == scene_id)
                        & (self.pixel_detections["chip_index"] == cur_chip_index)
                        & (self.pixel_detections["vessel_class"] != BACKGROUND)
                    ]

                    for _, detect in detects.iterrows():
                        if self.use_box_labels:
                            width = detect.right - detect.left
                            height = detect.bottom - detect.top
                            xmin = off_col + detect.columns - width/2
                            xmax = off_col + detect.columns + width/2
                            ymin = off_row + detect.rows - height/2
                            ymax = off_row + detect.rows + height/2
                        else:
                            xmin = off_col + detect.columns - self.bbox_size
                            xmax = off_col + detect.columns + self.bbox_size
                            ymin = off_row + detect.rows - self.bbox_size
                            ymax = off_row + detect.rows + self.bbox_size

                        centers.append([off_col + detect.columns, off_row + detect.rows])
                        boxes.append([xmin, ymin, xmax, ymax])
                        length_labels.append(detect.vessel_length_m)
                        score_labels.append(detect.score)

                        if detect.is_fishing == True:
                            class_labels.append(FISHING)
                        elif detect.is_vessel == True:
                            class_labels.append(NONFISHING)
                        else:
                            class_labels.append(NONVESSEL)

                        if detect.confidence == 'HIGH':
                            confidence_labels.append(2)
                        elif detect.confidence == 'MEDIUM':
                            confidence_labels.append(1)
                        elif detect.confidence == 'LOW':
                            confidence_labels.append(0)
                        else:
                            confidence_labels.append(-1)

                        if detect.is_vessel == True and detect.is_fishing == True:
                            fishing_labels.append(1)
                        elif detect.is_vessel == True and detect.is_fishing == False:
                            fishing_labels.append(0)
                        else:
                            fishing_labels.append(-1)

                        if detect.is_vessel == True:
                            vessel_labels.append(1)
                        elif detect.is_vessel == False:
                            vessel_labels.append(0)
                        else:
                            vessel_labels.append(-1)

            if self.class_map:
                class_labels = [self.class_map[cls-1] for cls in class_labels]

            if len(boxes) == 0:
                centers = torch.zeros((0, 2), dtype=torch.float32)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                class_labels = torch.zeros((1,), dtype=torch.int64)
                length_labels = torch.zeros((0,), dtype=torch.float32)
                confidence_labels = torch.zeros((0,), dtype=torch.int64)
                fishing_labels = torch.zeros((0,), dtype=torch.int64)
                vessel_labels = torch.zeros((0,), dtype=torch.int64)
                score_labels = torch.zeros((0,), dtype=torch.float32)
            else:
                centers = torch.as_tensor(centers, dtype=torch.float32)
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                class_labels = torch.as_tensor(class_labels, dtype=torch.int64)
                length_labels = torch.as_tensor(length_labels, dtype=torch.float32)
                confidence_labels = torch.as_tensor(confidence_labels, dtype=torch.int64)
                fishing_labels = torch.as_tensor(fishing_labels, dtype=torch.int64)
                vessel_labels = torch.as_tensor(vessel_labels, dtype=torch.int64)
                score_labels = torch.as_tensor(score_labels, dtype=torch.float32)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            # Return dummy values for inference
            centers = torch.tensor([])
            boxes = torch.tensor([])
            class_labels = torch.tensor(-1)
            length_labels = torch.tensor(-1)
            area = torch.tensor(-1)
            confidence_labels = torch.tensor(-1)

        # Create target dictionary in expected format for Faster R-CNN
        target = {}
        target["centers"] = centers
        target["boxes"] = boxes
        target["labels"] = class_labels
        target["length_labels"] = length_labels
        target["confidence_labels"] = confidence_labels
        target["fishing_labels"] = fishing_labels
        target["vessel_labels"] = vessel_labels
        target["score_labels"] = score_labels
        target["scene_id"] = scene_id
        target["chip_id"] = torch.tensor(chip_index)
        target["image_id"] = torch.tensor(idx)
        target["area"] = area
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        target["confidence"] = confidence_labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.clip_boxes:
            # Clip to image.
            target['boxes'] = torch.stack([
                torch.clip(target['boxes'][:, 0], min=0, max=img.shape[2]),
                torch.clip(target['boxes'][:, 1], min=0, max=img.shape[1]),
                torch.clip(target['boxes'][:, 2], min=0, max=img.shape[2]),
                torch.clip(target['boxes'][:, 3], min=0, max=img.shape[1]),
            ], dim=1)

        return img, target

    def get_chip_number(self, scene_id):
        """
        Get number of chips using first channel
        """
        return len(glob.glob(f"{self.chips_path}/{scene_id}/{self.channels[0]}/*.npy"))

    def add_background_chips(self):
        """
        Add background chips with no detections
        """
        for scene_id in self.scenes:
            # getting chip number for scene
            num_chips = self.get_chip_number(scene_id)

            # getting chips that have detections
            scene_detect_chips = (
                self.pixel_detections[self.pixel_detections["scene_id"] == scene_id][
                    "chip_index"
                ]
                .astype(int)
                .tolist()
            )

            # getting chips that do not have any detections
            scene_background_chips = [
                a for a in range(num_chips) if a not in list(set(scene_detect_chips))
            ]

            # eliminate chips that don't exist on disk
            scene_disk_chips = get_valid_chips(self.channels, os.path.join(self.chips_path, scene_id))
            scene_background_chips = [
                a for a in scene_background_chips if a in scene_disk_chips
            ]

            # computing the number of chips required
            num_background = int(
                self.background_frac * max(len(scene_detect_chips), self.background_min)
            )
            num_background = min(num_background, len(scene_background_chips))

            # obtaining a random set of chips without detections as background;
            # adding rows to a dataframe that will be appended to the dataset's
            # pixel_detections field
            np.random.seed(seed=0)
            chip_nums = np.random.choice(
                scene_background_chips, size=num_background, replace=False
            )
            rows = []
            cols = [
                "index",
                "detect_lat",
                "detect_lon",
                "vessel_length_m",
                "source",
                "detect_scene_row",
                "detect_scene_column",
                "is_vessel",
                "is_fishing",
                "distance_from_shore_km",
                "scene_id",
                "confidence",
                "top",
                "left",
                "bottom",
                "right",
                "detect_id",
                "vessel_class",
                "scene_rows",
                "scene_cols",
                "rows",
                "columns",
                "chip_index",
                "score",
            ]
            for ii in range(num_background):
                row = [
                    -1,  #'index'
                    -1,  #'detect_lat'
                    -1,  # 'detect_lon'
                    -1,  #'vessel_length_m'
                    "background",  # source
                    -1,  #'detect_scene_row',
                    -1,  #'detect_scene_column',
                    -1,  #'is_vessel'
                    -1,  #'is_fishing',
                    -1,  #'distance_from_shore_km',
                    scene_id,  #'scene_id',
                    -1, #'confidence',
                    -1,  # top
                    -1,  # left
                    -1,  # bottom
                    -1,  # right
                    -1,  #'detect_id',
                    BACKGROUND,  #'vessel_class',
                    -1,  #'scene_rows',
                    -1,  #'scene_cols',
                    -1,  #'rows',
                    -1,  #'columns',
                    chip_nums[ii],  # chip_index
                    1.0,
                ]

                rows.append(row)
            # Background chips for this scene to dataframe
            df_background = pd.DataFrame(rows, columns=cols)
            # Append background chips for this scene to dataset-level detections
            # dataframe
            self.pixel_detections = pd.concat((self.pixel_detections, df_background))

    def chip_and_get_pixel_detections(self):
        """
        Load all preprocessed scene dataframes
        """
        try:
            all_chip_detections = pd.read_csv(self.annotation_path)
            print("Loaded {} detections from {}".format(len(all_chip_detections), self.annotation_path))
        except:
            print(f"No detections found at  {self.annotation_path}, proceeding without")
            return None

        # Add new columns.
        if 'score' not in all_chip_detections.columns:
            all_chip_detections.insert(len(all_chip_detections.columns), 'score', [1.0]*len(all_chip_detections))

        # Ensure the scenes loaded align with the object list of scenes to use
        scene_detects = []
        print(f"Loading detections for scene {len(self.scenes)} scenes...")
        for jj, scene_id in enumerate(self.scenes):
            this_scene_detections = all_chip_detections[
                (all_chip_detections["scene_id"] == scene_id)
            ]
            chips_with_data = get_valid_chips(self.channels, os.path.join(self.chips_path, scene_id))
            valid_detects = this_scene_detections[this_scene_detections.chip_index.isin(chips_with_data)]

            scene_detects.append(valid_detects)

        pixel_detections = pd.concat(scene_detects).reset_index()


        return pixel_detections

    def get_geo_balanced_sampler(self):
        """
        Returns a torch.utils.data.Sampler that samples uniformly over space.
        """
        # Bucket all chips based on the 1/10 lat/lon that they fall into.
        # 800 pixel chip size is within a factor of 2 of 1/10 lat/lon.
        buckets = {}
        for i, (scene_id, chip_index) in enumerate(self.chip_indices):
            corners = get_chip_corners(self.chips_path, scene_id)[chip_index]
            bucket_x = int(math.floor(corners[0]*10))
            bucket_y = int(math.floor(corners[1]*10))
            k = (bucket_x, bucket_y)
            if k not in buckets:
                buckets[k] = []
            buckets[k].append(i)

        weights = [None]*len(self.chip_indices)
        for l in buckets.values():
            for i in l:
                weights[i] = 1.0/len(l)

        print('using geo_balanced_sampler with {} chips and {} buckets'.format(len(self.chip_indices), len(buckets)))
        return torch.utils.data.WeightedRandomSampler(weights, len(self.chip_indices))

    def get_bg_balanced_sampler(self, background_frac=1.0, incl_low_score=True, only_val_bg=False):
        """
        Returns a torch.utils.data.Sampler that samples foreground/background at 1:1 ratio.
        (Intended to function with AllChips=True.)
        """
        is_bg_map = {(scene_id, chip_index): True for scene_id, chip_index in self.chip_indices}
        for _, label in self.pixel_detections.iterrows():
            if label.vessel_class == BACKGROUND:
                continue
            if not incl_low_score and label.score < 1.0:
                continue
            is_bg_map[(label.scene_id, label.chip_index)] = False

        bg_indices = []
        fg_indices = []
        for i, (scene_id, chip_index) in enumerate(self.chip_indices):
            if not is_bg_map[(scene_id, chip_index)]:
                fg_indices.append(i)
            elif only_val_bg and not scene_id.endswith('v'):
                # Ignore background scenes not in validation set.
                pass
            else:
                bg_indices.append(i)

        bg_weight = background_frac * (len(fg_indices)/len(bg_indices))
        weights = [0.0]*len(self.chip_indices)
        for i in fg_indices:
            weights[i] = 1.0
        for i in bg_indices:
            weights[i] = bg_weight

        print('using bg_balanced_sampler with {} bg chips, {} fg chips (bg_weight={})'.format(len(bg_indices), len(fg_indices), bg_weight))
        return torch.utils.data.WeightedRandomSampler(weights, len(self.chip_indices))

    def get_i2(self, scene_id, chip_index, count=4):
        if scene_id not in self.i2_option_cache:
            with open(os.path.join(self.chips_path, scene_id, 'intersect2.json'), 'r') as f:
                self.i2_option_cache[scene_id] = json.load(f)

        options = self.i2_option_cache[scene_id][chip_index]
        # only use validation scenes if there's at least one of them
        if any([t[0].endswith('v') for t in options]):
            options = [t for t in options if t[0].endswith('v')]
        if len(options) > count:
            options = random.sample(options, count)
        im = np.zeros((12, 800, 800), dtype='float32')
        options = [(scene_id, self.chip_offsets[scene_id][chip_index][0], self.chip_offsets[scene_id][chip_index][1])] # DELETE ME
        for option_idx, (other_scene_id, col_offset, row_offset) in enumerate(options):
            if other_scene_id not in self.chip_offsets:
                with open(os.path.join(self.chips_path, other_scene_id, 'coords.json'), 'r') as f:
                    self.chip_offsets[other_scene_id] = [(col, row) for col, row in json.load(f)['offsets']]

            #col_offset += random.randint(-32, 32)
            #row_offset += random.randint(-32, 32)
            for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                cur_col = 800*(col_offset//800+i)
                cur_row = 800*(row_offset//800+j)

                if (cur_col, cur_row) not in self.chip_offsets[other_scene_id]:
                    continue

                other_chip_index = self.chip_offsets[other_scene_id].index((cur_col, cur_row))

                col_overlap = 800 - abs(col_offset - cur_col)
                row_overlap = 800 - abs(row_offset - cur_row)
                src_col_offset = max(col_offset - cur_col, 0)
                src_row_offset = max(row_offset - cur_row, 0)
                dst_col_offset = max(cur_col - col_offset, 0)
                dst_row_offset = max(cur_row - row_offset, 0)

                vh_path = os.path.join(self.chips_path, other_scene_id, 'vh/{}_vh.npy'.format(other_chip_index))
                vv_path = os.path.join(self.chips_path, other_scene_id, 'vv/{}_vv.npy'.format(other_chip_index))
                if not os.path.exists(vh_path) or not os.path.exists(vv_path):
                    continue

                vh = np.load(vh_path)
                vv = np.load(vv_path)
                vh = np.clip(vh+50, 0, 70)/70
                vv = np.clip(vv+50, 0, 70)/70
                im[3*option_idx+0, dst_row_offset:dst_row_offset+row_overlap, dst_col_offset:dst_col_offset+col_overlap] = vh[src_row_offset:src_row_offset+row_overlap, src_col_offset:src_col_offset+col_overlap]
                im[3*option_idx+1, dst_row_offset:dst_row_offset+row_overlap, dst_col_offset:dst_col_offset+col_overlap] = vv[src_row_offset:src_row_offset+row_overlap, src_col_offset:src_col_offset+col_overlap]

            detects = self.pixel_detections[
                (self.pixel_detections["scene_id"] == other_scene_id)
                & (self.pixel_detections["detect_scene_row"] >= row_offset)
                & (self.pixel_detections["detect_scene_column"] >= col_offset)
                & (self.pixel_detections["detect_scene_row"] < row_offset+800)
                & (self.pixel_detections["detect_scene_column"] < col_offset+800)
                & (self.pixel_detections["vessel_class"] != BACKGROUND)
            ]

            for _, detect in detects.iterrows():
                row = detect.detect_scene_row - row_offset
                col = detect.detect_scene_column - col_offset
                if detect.confidence == 'LOW':
                    im[3*option_idx+2, row, col] = 0.5
                else:
                    im[3*option_idx+2, row, col] = 1.0
        return im
