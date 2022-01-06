import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree, distance_matrix
from tqdm import tqdm
import sys

sys.path.insert(1, '/home/xview3/src') # use an appropriate path if not in the docker volume

from xview3.processing.constants import PIX_TO_M

def drop_low_confidence_preds(pred, gt, distance_tolerance=200, costly_dist=False):
    """
    Matches detections in a predictions dataframe to a ground truth data frame and isolate the low confidence matches

    Args:
        preds (pd.DataFrame): contains inference results for a
            single scene
        gt (pd.DataFrame): contains ground truth labels for a single
            scene
        distance_tolerance (int, optional): Maximum distance
            for valid detection. Defaults to 200.
        costly_dist (bool): whether to assign 9999999 to entries in the
            distance metrics greater than distance_tolerance; defaults to False

    Returns:
        df_out (pd.DataFrame): preds dataframe without the low confidence matches
    """

    low_inds = []

    # For each scene, obtain the tp, fp, and fn indices for maritime
    # object detection in the *global* pred and gt dataframes
    for scene_id in tqdm(gt["scene_id"].unique()):
        pred_sc = pred[pred["scene_id"] == scene_id]
        if len(pred_sc) == 0:
            continue
        gt_sc = gt[gt["scene_id"] == scene_id]
        low_inds_scene = match_low_confidence_preds(
            pred_sc, gt_sc, distance_tolerance=distance_tolerance, costly_dist=costly_dist
        )

        low_inds += low_inds_scene

    # Check matched pairs came from "LOW" labels
    for pair in low_inds:
        assert gt.iloc[pair["gt_idx"]]["confidence"] == "LOW", f"Index {pair['gt_idx']} is {gt.iloc[pair['gt_idx']]['confidence']}"

    low_pred_inds = [a["pred_idx"] for a in low_inds]

    df_out = pred.drop(index=low_pred_inds)
    df_out = df_out.reset_index()
    return df_out

def match_low_confidence_preds(preds, gt, distance_tolerance=200, costly_dist=False):
    """
    Matches detections in a predictions dataframe to a ground truth data frame and isolate the low confidence matches

    Args:
        preds (pd.DataFrame): contains inference results for a
            single scene
        gt (pd.DataFrame): contains ground truth labels for a single
            scene
        distance_tolerance (int, optional): Maximum distance
            for valid detection. Defaults to 200.
        costly_dist (bool): whether to assign 9999999 to entries in the
            distance metrics greater than distance_tolerance; defaults to False

    Returns:
        low_inds (list, int): list of indices for the preds dataframe that are
            associated as (1) correct detection in the *global* preds dataframe; (2) low confidence in the corresponding gt dataframe
    """

    # Getting pixel-level predicted and ground-truth detections
    pred_array = np.array(
        list(zip(preds["detect_scene_row"], preds["detect_scene_column"]))
    )
    gt_array = np.array(list(zip(gt["detect_scene_row"], gt["detect_scene_column"])))

    # Getting a list of index with LOW in the ground truth dataframe
    low_gt_inds = list(gt[gt["confidence"] == "LOW"].index)

    # Building distance matrix using Euclidean distance pixel space
    # multiplied by the UTM resolution (10 m per pixel)
    dist_mat = distance_matrix(pred_array, gt_array, p=2) * PIX_TO_M
    if costly_dist:
        dist_mat[dist_mat > distance_tolerance] = 9999999 * PIX_TO_M

    # Using Hungarian matching algorithm to assign lowest-cost gt-pred pairs
    rows, cols = linear_sum_assignment(dist_mat)

    low_inds = [
        {"pred_idx": preds.index[rows[ii]], "gt_idx": gt.index[cols[ii]]}
        for ii in range(len(rows))
        if (dist_mat[rows[ii], cols[ii]] < distance_tolerance) and (gt.index[cols[ii]] in low_gt_inds)
    ]

    return low_inds

def get_shore_preds(df, shoreline_root, scene_id, shore_tolerance_km):
    """
    Getting detections that are close to the shoreline

    Args:
        df (pd.DataFrame): dataframe containing detections
        shoreline_root (str): path to shoreline contour files
        scene_id (str): scene_id
        shore_tolerance_km (float): "close to shore" tolerance in km

    Returns:
        df_close (pd.DataFrame): subset of df containing only detections close to shore
    """
    # Loading shoreline contours for distance-to-shore calculation
    shoreline_contours = np.load(
        f"{shoreline_root}/{scene_id}_shoreline.npy", allow_pickle=True
    )

    # If there are no shorelines in the scene
    if len(shoreline_contours) == 0:
        return pd.DataFrame()

    contour_points = np.vstack(shoreline_contours)

    # Creating KD trees and computing distance matrix
    tree1 = KDTree(np.array(contour_points))
    tree2 = KDTree(
        np.array([df["detect_scene_row"], df["detect_scene_column"]]).transpose()
    )
    sdm = tree1.sparse_distance_matrix(tree2, shore_tolerance_km * 1000 / PIX_TO_M, p=2)
    dists = sdm.toarray()

    # Make it so we can use np.min() to find smallest distance b/t each detection and any contour point
    dists[dists == 0] = 9999999
    min_shore_dists = np.min(dists, axis=0)
    close_shore_inds = np.where(min_shore_dists != 9999999)
    df_close = df.iloc[close_shore_inds]
    return df_close


def compute_loc_performance(preds, gt, distance_tolerance=200, costly_dist=False):
    """
    Computes maritime object detection performance from a prediction
    dataframe and a ground truth datafr

    Args:
        preds (pd.DataFrame): contains inference results for a
            single scene
        gt (pd.DataFrame): contains ground truth labels for a single
            scene
        distance_tolerance (int, optional): Maximum distance
            for valid detection. Defaults to 200.
        costly_dist (bool): whether to assign 9999999 to entries in the distance metrics greater than distance_tolerance; defaults to False

    Returns:
        tp_ind (list, dict): list of dicts with keys 'pred_idx', 'gt_idx';
            values for each are the indices preds and gt that are
            associated as correct detection in the *global* preds and
            gt dataframes
        fp_ind (list): list of indices in the *global* preds dataframe
            that are not assigned to a gt detection by the matching
        fn_ind (list): list of indices in the *global* gt dataframe
            that do not match any detection in pred within dist_tol
    """
    # distance_matrix below doesn't work when preds is empty, so handle that first
    if len(preds) == 0:
        return [], [], [a for a in gt.index]

    # Getting pixel-level predicted and ground-truth detections
    pred_array = np.array(
        list(zip(preds["detect_scene_row"], preds["detect_scene_column"]))
    )
    gt_array = np.array(list(zip(gt["detect_scene_row"], gt["detect_scene_column"])))

    # Building distance matrix using Euclidean distance pixel space
    # multiplied by the UTM resolution (10 m per pixel)
    dist_mat = distance_matrix(pred_array, gt_array, p=2) * PIX_TO_M
    if costly_dist:
        dist_mat[dist_mat > distance_tolerance] = 9999999 * PIX_TO_M

    # Using Hungarian matching algorithm to assign lowest-cost gt-pred pairs
    rows, cols = linear_sum_assignment(dist_mat)

    # Recording indices for tp, fp, fn
    tp_inds = [
        {"pred_idx": preds.index[rows[ii]], "gt_idx": gt.index[cols[ii]]}
        for ii in range(len(rows))
        if dist_mat[rows[ii], cols[ii]] < distance_tolerance
    ]
    tp_pred_inds = [a["pred_idx"] for a in tp_inds]
    tp_gt_inds = [a["gt_idx"] for a in tp_inds]

    fp_inds = [a for a in preds.index if a not in tp_pred_inds]
    fn_inds = [a for a in gt.index if a not in tp_gt_inds]

    # Making sure each GT is associated with one true positive
    # or is in the false negative bin
    assert len(gt) == len(fn_inds) + len(tp_inds)

    return tp_inds, fp_inds, fn_inds


def compute_vessel_class_performance(preds, gt, tp_inds):
    """
    Identify tp, tn, fp, and fn indices for vessel classification task

    Args:
        preds ([pd.Dataframe)): contains inference results for all scenes
        gt (pd.Dataframe): contains ground truth labels for all scenes
        tp_inds (list, dict): List of dicts output from compute_loc_performance
            containing indices of true positive detection pairs in the global preds
            and gt dataframes

    Returns:
        c_tp_inds: list of dicts from tp_inds where ground truth and
            model output both have a True 'is_vessel' label
        c_tn_inds: list of dicts from tp_inds where ground truth and
            model output both have a False 'is_vessel' label
        c_fp_inds: list of dicts from tp_inds where ground truth
            'is_vessel' label is False but model output is True
        c_fn_inds: list of dicts from tp_inds where ground truth
            'is_vessel' output is True but model output is False
    """
    c_tp_inds = []
    c_fp_inds = []
    c_fn_inds = []
    c_tn_inds = []
    # For every box where you have a matching detection, do the labels match?
    for pair in tp_inds:
        # Making sure we only use valid GTs
        if isinstance(gt[pair["gt_idx"]], float):
            if np.isnan(gt[pair["gt_idx"]]):
                continue
        if preds[pair["pred_idx"]] == gt[pair["gt_idx"]]:
            if gt[pair["gt_idx"]]:
                c_tp_inds.append(pair)
            else:
                c_tn_inds.append(pair)
        else:
            if gt[pair["gt_idx"]]:
                c_fn_inds.append(pair)
            elif gt[pair["gt_idx"]] == False:
                c_fp_inds.append(pair)

    return c_tp_inds, c_fp_inds, c_fn_inds, c_tn_inds


def compute_fishing_class_performance(preds, gt, tp_inds, vessel_inds):
    """
    Identify tp, tn, fp, and fn indices for fishing classification task

    Args:
        preds ([pd.Dataframe)): contains inference results for all scenes
        gt (pd.Dataframe): contains ground truth labels for all scenes
        tp_inds (list, dict): List of dicts output from compute_loc_performance
            containing indices of true positive detection pairs in the global preds
            and gt dataframes

    Returns:
        c_tp_inds: list of dicts from tp_inds where ground truth and
            model output both have a True 'is_fishing' label
        c_tn_inds: list of dicts from tp_inds where ground truth and
            model output both have a False 'is_fishing' label
        c_fp_inds: list of dicts from tp_inds where ground truth
            'is_fishing' label is False but model output is True
        c_fn_inds: list of dicts from tp_inds where ground truth
            'is_fishing' output is True but model output is False
    """
    c_tp_inds = []
    c_fp_inds = []
    c_fn_inds = []
    c_tn_inds = []
    # For every box where you have a matching detection, do the labels match?
    for pair in tp_inds:
        if vessel_inds is not None:
            if not vessel_inds[pair["gt_idx"]]:
                # print('Skipping non-vessels')
                continue
        # Making sure we only use valid GTs
        if isinstance(gt[pair["gt_idx"]], float):
            if np.isnan(gt[pair["gt_idx"]]):
                continue
        if preds[pair["pred_idx"]] == gt[pair["gt_idx"]]:
            if gt[pair["gt_idx"]]:
                c_tp_inds.append(pair)
            else:
                c_tn_inds.append(pair)
        else:
            if gt[pair["gt_idx"]]:
                c_fn_inds.append(pair)
            elif gt[pair["gt_idx"]] == False:
                c_fp_inds.append(pair)

    return c_tp_inds, c_fp_inds, c_fn_inds, c_tn_inds


def compute_length_performance(preds, gt, tp_inds):
    """
    Compute aggregate percent error for vessel size estimation

    Args:
        preds ([pd.Dataframe)): contains inference results for all scenes
        gt (pd.Dataframe): contains ground truth labels for all scenes
        tp_inds (list, dict): List of dicts output from compute_loc_performance
            containing indices of true positive detection pairs in the global preds
            and gt dataframes

    Returns:
        length_performance (float): aggregate percent error for vessel length estimation
    """
    pct_error = 0.0
    num_valid_gt = 0.0

    for pair in tp_inds:
        if isinstance(gt[pair["gt_idx"]], float):
            if np.isnan(gt[pair["gt_idx"]]):
                continue
        pct_error += (
            np.abs(preds[pair["pred_idx"]] - gt[pair["gt_idx"]]) / gt[pair["gt_idx"]]
        )
        # TODO: update this -- make it min(pct_error, 1.0)
        num_valid_gt += 1

    if num_valid_gt == 0:
        length_performance = 0
    else:
        length_performance = 1.0 - min((pct_error / num_valid_gt), 1.0)

    return length_performance


def calculate_p_r_f(num_tp, num_fp, num_fn):
    """
    Compute precision, recall, and F1 score

    Args:
        tp_inds (list, dict): list of dicts with keys 'pred_idx', 'gt_idx';
            values for each are the indices preds and gt that are
            associated as correct detection in the *global* preds and
            gt dataframes
        fp_inds (list): list of indices in the *global* preds dataframe
            that are not assigned to a gt detection by the matching
        fn_inds (list): list of indices in the *global* gt dataframe
            that do not match any detection in pred within dist_tol

    Returns:
        precision (float): precision score
        recall (float): recall score
        fscore (float): f1 score
    """
    try:
        precision = num_tp / (num_tp + num_fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = num_tp / (num_tp + num_fn)
    except ZeroDivisionError:
        recall = 0
    try:
        fscore = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        fscore = 0

    if precision == np.nan or recall == np.nan or fscore == np.nan:
        return 0, 0, 0
    else:
        return precision, recall, fscore


def aggregate_f(
    loc_fscore, length_acc, vessel_fscore, fishing_fscore, loc_fscore_shore
):
    """
    Compute aggregate metric for xView3 scoring


    Args:
        loc_fscore (float): F1 score for overall maritime object detection
        length_acc (float): Aggregate percent error for vessel length estimation
        vessel_fscore (float): F1 score for vessel vs. non-vessel task
        fishing_fscore (float): F1 score for fishing vessel vs. non-fishing vessel task
        loc_fscore_shore (float): F1 score for close-to-shore maritime object detection

    Returns:
        aggregate (float): aggregate metric for xView3 scoring
    """

    # Note: should be between zero and one, and score should be heavily weighted on
    # overall maritime object detection!
    aggregate = (
        loc_fscore
        * (1 + length_acc + vessel_fscore + fishing_fscore + loc_fscore_shore)
        / 5
    )

    return aggregate


def score(pred, gt, shore_root, distance_tolerance=200, shore_tolerance=2, quiet=False, weights_fname=None, costly_dist=False):
    """Compute xView3 aggregate score from

    Args:
        pred ([pd.Dataframe)): contains inference results for all scenes
        gt (pd.Dataframe): contains ground truth labels for all scenes]
        shoreline_root (str): path to shoreline contour files
        distance_tolerance (float): Maximum distance
            for valid detection. Defaults to 200.
        shore_tolerance (float): "close to shore" tolerance in km; defaults to 2
        costly_dist (bool): whether to assign 9999999 to entries in the distance metrics greater than distance_tolerance; defaults to False

    Returns:
        scores (dict): dictionary containing aggregate xView score and
            all constituent scores
    """
    scores = {}

    tp_inds, fp_inds, fn_inds = [], [], []
    num_tp, num_fp, num_fn = 0, 0, 0

    if weights_fname:
        with open(weights_fname, 'r') as f:
            weights = json.load(f)
    else:
        weights = {scene_id: 1 for scene_id in gt["scene_id"].unique()}

    # For each scene, obtain the tp, fp, and fn indices for maritime
    # object detection in the *global* pred and gt dataframes
    for scene_id in gt["scene_id"].unique():
        pred_sc = pred[pred["scene_id"] == scene_id]
        gt_sc = gt[gt["scene_id"] == scene_id]
        tp_inds_sc, fp_inds_sc, fn_inds_sc, = compute_loc_performance(
            pred_sc, gt_sc, distance_tolerance=distance_tolerance, costly_dist=costly_dist
        )

        tp_inds += tp_inds_sc
        fp_inds += fp_inds_sc
        fn_inds += fn_inds_sc

        num_tp += weights[scene_id]*len(tp_inds_sc)
        num_fp += weights[scene_id]*len(fp_inds_sc)
        num_fn += weights[scene_id]*len(fn_inds_sc)

    # Compute precision, recall, and F1 for maritime object detection
    loc_precision, loc_recall, loc_fscore = calculate_p_r_f(num_tp, num_fp, num_fn)

    # Allowing code to be run without shore data -- will output 0 for these scores
    if shore_tolerance and shore_root:
        # For each scene, compute distances to shore for model predictions, and isolate
        # both predictions and ground truth that are within the appropriate distance
        # from shore.  Note that for the predictions, we include any predictions within
        # shore_tolerance + distance_tolerance/1000.
        tp_inds_shore, fp_inds_shore, fn_inds_shore = [], [], []
        num_tp_shore, num_fp_shore, num_fn_shore = 0, 0, 0
        for scene_id in gt["scene_id"].unique():
            pred_sc = pred[pred["scene_id"] == scene_id]
            gt_sc_shore = gt[
                (gt["scene_id"] == scene_id)
                & (gt["distance_from_shore_km"] <= shore_tolerance)
            ]
            pred_sc_shore = get_shore_preds(
                pred_sc,
                shore_root,
                scene_id,
                shore_tolerance + distance_tolerance / 1000,
            )
            if not quiet:
                print(
                    f"{len(gt_sc_shore)} ground truth, {len(pred_sc_shore)} predictions close to shore"
                )
            # For each scene, compute tp, fp, fn indices by applying the matching algorithm
            # while only considering close-to-shore predictions and detections
            if (len(gt_sc_shore) > 0) and (len(pred_sc_shore) > 0):
                (
                    tp_inds_sc_shore,
                    fp_inds_sc_shore,
                    fn_inds_sc_shore,
                ) = compute_loc_performance(
                    pred_sc_shore, gt_sc_shore, distance_tolerance=distance_tolerance, costly_dist=costly_dist
                )
                tp_inds_shore += tp_inds_sc_shore
                fp_inds_shore += fp_inds_sc_shore
                fn_inds_shore += fn_inds_sc_shore
                num_tp_shore += weights[scene_id]*len(tp_inds_sc_shore)
                num_fp_shore += weights[scene_id]*len(fp_inds_sc_shore)
                num_fn_shore += weights[scene_id]*len(fn_inds_sc_shore)

        if (
            len(
                gt[
                    (gt["scene_id"].isin(list(pred["scene_id"].unique())))
                    & (gt["distance_from_shore_km"] <= shore_tolerance)
                ]
            )
            > 0
        ):
            # Compute precision, recall, F1 for close-to-shore maritime object detection
            loc_precision_shore, loc_recall_shore, loc_fscore_shore = calculate_p_r_f(
                num_tp_shore, num_fp_shore, num_fn_shore
            )
        else:
            loc_precision_shore, loc_recall_shore, loc_fscore_shore = 0, 0, 0
    else:
        loc_precision_shore, loc_recall_shore, loc_fscore_shore = 0, 0, 0

    # Getting ground truth vessel indices using is_vessel field in gt
    vessel_inds = gt["is_vessel"].isin([True])

    # Getting performance on vessel classification task
    v_tp_inds, v_fp_inds, v_fn_inds, v_tn_inds = compute_vessel_class_performance(
        pred["is_vessel"].values, gt["is_vessel"].values, tp_inds
    )
    vessel_precision, vessel_recall, vessel_fscore = calculate_p_r_f(
        len(v_tp_inds),
        len(v_fp_inds),
        len(v_fn_inds),
    )

    # Getting performance on fishing classification; note that we only consider
    # ground-truth detections that are actually vessels
    f_tp_inds, f_fp_inds, f_fn_inds, f_tn_inds = compute_fishing_class_performance(
        pred["is_fishing"].values, gt["is_fishing"].values, tp_inds, vessel_inds
    )
    fishing_precision, fishing_recall, fishing_fscore = calculate_p_r_f(
        len(f_tp_inds),
        len(f_fp_inds),
        len(f_fn_inds),
    )

    # Computing length estimation performance
    inf_lengths = pred["vessel_length_m"].tolist()
    gt_lengths = gt["vessel_length_m"].tolist()
    length_acc = compute_length_performance(inf_lengths, gt_lengths, tp_inds)

    # Computing normalized aggregate metric
    aggregate = aggregate_f(
        loc_fscore, length_acc, vessel_fscore, fishing_fscore, loc_fscore_shore
    )

    # Creating score dictionary
    scores = {
        "loc_fscore": loc_fscore,
        "loc_fscore_shore": loc_fscore_shore,
        "vessel_fscore": vessel_fscore,
        "fishing_fscore": fishing_fscore,
        "length_acc": length_acc,
        "aggregate": aggregate,
        "loc_precision": loc_precision,
        "loc_recall": loc_recall,
        "loc_precision_shore": loc_precision_shore,
        "loc_recall_shore": loc_recall_shore,
    }

    # Metadata that's useful for creating visualizations of detection performance.
    get_pos = lambda a: [
        a['scene_id'],
        int(a['detect_scene_row']),
        int(a['detect_scene_column']),
        bool(a['is_vessel'] == True),
        bool(a['is_fishing'] == True),
        a.get('confidence', None),
        a.get('source', None),
    ]
    meta = {
        'tp': [(get_pos(gt.loc[a['gt_idx']]), get_pos(pred.loc[a['pred_idx']])) for a in tp_inds],
        'fp': [get_pos(pred.loc[a]) for a in fp_inds],
        'fn': [get_pos(gt.loc[a]) for a in fn_inds],
    }

    return scores, meta


def main(args):
    print(f"--score_all: {args.score_all}")
    print(f"--costly_dist: {args.costly_dist}")
    print(f"--drop_low_detect: {args.drop_low_detect}")

    # Read in inference and ground truth detection files
    inference = pd.read_csv(args.inference_file, index_col=False)
    ground_truth = pd.read_csv(args.label_file, index_col=False)

    if args.bathymetry_threshold is not None:
        inference = inference[inference["bathymetry"] < args.bathymetry_threshold]

    # If a scene_id list is provided, run only for that scene; otherwise,
    # use all scenes in ground truth
    if args.scene_id is not None:
        inference = inference[inference["scene_id"] == args.scene_id]
        ground_truth = ground_truth[
            ground_truth["scene_id"] == args.scene_id
        ]
    elif args.scene_path is not None:
        with open(args.scene_path, 'r') as f:
            scene_ids = [line.strip() for line in f.readlines() if line.strip()]
        ground_truth = ground_truth[
            ground_truth["scene_id"].isin(scene_ids)
        ]

    if args.fishing_threshold:
        inference['is_fishing'] = (inference['vessel_score'] >= args.vessel_threshold) & (inference['fishing_score'] >= args.fishing_threshold)
        inference['is_vessel'] = inference['vessel_score'] >= args.vessel_threshold

    if args.cls == 'fishing':
        inference = inference[inference.is_fishing == True]
        ground_truth = ground_truth[ground_truth.is_fishing == True]
    elif args.cls == 'vessel':
        inference = inference[inference.is_vessel == True]
        ground_truth = ground_truth[ground_truth.is_vessel == True]
    elif args.cls == 'object':
        inference = inference[inference.is_vessel == False]
        ground_truth = ground_truth[ground_truth.is_vessel == False]

    inference = inference.reset_index(drop=True)
    ground_truth = ground_truth.reset_index(drop=True)

    # By default we only score on high and medium confidence labels
    if not args.score_all:
        if args.drop_low_detect:
            inference = drop_low_confidence_preds(inference, ground_truth, distance_tolerance=args.distance_tolerance)
        ground_truth = ground_truth[
            ground_truth["confidence"].isin(["HIGH", "MEDIUM"])
        ]

    inference = inference.reset_index(drop=True)
    ground_truth = ground_truth.reset_index(drop=True)

    if args.threshold < 0:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        thresholds = [args.threshold]

    best = None
    for threshold in thresholds:
        if threshold > 0:
            cur_pred = inference[inference["score"] >= threshold].reset_index(drop=True)
        else:
            cur_pred = inference
        if len(cur_pred) == 0:
            continue
        out, meta = score(
            cur_pred,
            ground_truth,
            args.shore_root,
            args.distance_tolerance,
            args.shore_tolerance,
            weights_fname=args.weights,
            costly_dist=args.costly_dist,
        )
        print(threshold, out)
        if best is None or out['loc_fscore'] > best[0]['loc_fscore']:
            best = (out, meta)

    print('best', best[0])
    if args.output:
        with open(args.output, "w") as fl:
            json.dump(best[0], fl)

    if args.meta:
        with open(args.meta, "w") as fl:
            json.dump(best[1], fl)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scoring xView3 model.")
    parser.add_argument(
        "--scene_id", help="Scene ID to run evaluations for", default=None
    )
    parser.add_argument("--scene_path", help="Path to the scene split list", default=None)
    parser.add_argument("--inference_file", help="Path to the predictions CSV")
    parser.add_argument("--label_file", help="Path to the xView3 label CSV")
    parser.add_argument("--output", help="Path to output file -- should be .json")
    parser.add_argument("--meta", default=None, help="Path to save visualization metadata (.json)")
    parser.add_argument("--weights", default=None, help="Path to JSON with scoring weights (dict from scene to weight)")
    parser.add_argument(
        "--distance_tolerance", help="Distance tolerance for detection in m", type=int, default=200,
    )
    parser.add_argument(
        "--shore_tolerance",
        default=2,
        help="Distance from shore tolerance in km",
        type=int,
    )
    parser.add_argument(
        "--shore_root",
        type=str,
        default=None,
        help="Directory with .npy files containing shore arrays",
    )
    parser.add_argument(
        "--score_all",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to score against all ground truth labels (inclusive of low confidence labels).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Only use predictions with score above this threshold, or -1 to try many thresholds"
    )
    parser.add_argument(
        "--fishing_threshold",
        type=float,
        default=None,
        help="Set is_fishing based on fishing_score with this threshold"
    )
    parser.add_argument(
        "--vessel_threshold",
        type=float,
        default=None,
        help="Set is_vessel based on vessel_score with this threshold"
    )
    parser.add_argument(
        "--bathymetry_threshold",
        type=float,
        default=None,
        help="Only use predictions with bathymetry below this threshold",
    )
    parser.add_argument(
        "--cls",
        default=None,
        help="Only compare predictions and labels in this class (fishing, vessel, or object)",
    )
    parser.add_argument(
        "--drop_low_detect",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to drop predictions that are matched to low confidence labels.",
    )
    parser.add_argument(
        "--costly_dist",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to assign a large number (9999999) to distances greater than the distance tolerance threshold.",
    )

    args = parser.parse_args()

    main(args)
