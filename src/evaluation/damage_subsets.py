"""Evaluate the BBCH predictions separately for plants with and without damage.
 
This module splits the predictions and the ground truth (GT) into two subsets
based on the damage_visible attribute of the GT plants, and prints the BBCH
detection metrics (per class F1, MAE and BIAS) for each subset. It is used by the
evaluation scripts and has no command line entry point.
 
Predictions are matched to GT plants by mask IoU. A prediction is put into a
subset if its matched GT plant is in that subset. Predictions without a match are
dropped from both subsets. GT plants without a damage_visible attribute are in
neither subset.
"""


import numpy as np
from pycocotools import mask as mask_utils
import warnings
from metrics import InstanceDetectionMetrics, LocalizationMetrics


def _masks(item):
    """Collect all mask annotations of an item.
 
    Returns a list with the mask annotations.
    """
    return [a for a in item.get("annotations", []) if a.get("type") == "mask"]


def _is_damaged(ann):
    """Read the damage_visible attribute of an annotation as a bool.
 
    String values like "true", "1" or "yes" count as True.
 
    Returns True or False, or None if the attribute is missing.
    """
    value = ann.get("attributes", {}).get("damage_visible")
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _greedy_pairs(pred_rles, gt_rles, iou_threshold):
    """Match predicted and GT masks one-to-one by mask IoU.
 
    The pairs with the highest IoU are matched first. Each mask is used at most
    once, and pairs below iou_threshold are not matched.
 
    Returns a list of (pred_index, gt_index) pairs.
    """
    if not pred_rles or not gt_rles:
        return []
    iou_matrix = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles))
    matched_p, matched_g, pairs = set(), set(), []
    for idx in np.argsort(iou_matrix, axis=None)[::-1]:
        p, g = np.unravel_index(idx, iou_matrix.shape)
        if p in matched_p or g in matched_g:
            continue
        if iou_matrix[p, g] < iou_threshold:
            break
        matched_p.add(p)
        matched_g.add(g)
        pairs.append((int(p), int(g)))
    return pairs


def split_by_damage(pred_data, gt_data, damaged, iou_threshold):
    """Build the prediction and GT subset for one damage state.
 
    The GT keeps only plants whose damage_visible equals `damaged`. The predictions
    keep only masks that are matched (see _greedy_pairs) to one of these GT plants.
    Prediction items without a GT item are skipped.
 
    Returns the prediction subset, the GT subset and the number of GT plants
    without a damage_visible attribute.
    """
    gt_by_id = {it["id"]: it for it in gt_data["items"]}
    pred_sub = {k: v for k, v in pred_data.items() if k != "items"}
    gt_sub = {k: v for k, v in gt_data.items() if k != "items"}
    pred_sub["items"], gt_sub["items"] = [], []
    n_unknown = 0

    for p_item in pred_data["items"]:
        g_item = gt_by_id.get(p_item["id"])
        if g_item is None:
            continue
        P, G = _masks(p_item), _masks(g_item)
        pairs = _greedy_pairs([a["rle"] for a in P], [a["rle"] for a in G], iou_threshold)

        flags = [_is_damaged(g) for g in G]
        n_unknown += sum(f is None for f in flags)
        keep_g = {i for i, f in enumerate(flags) if f is damaged}
        keep_p = sorted(p for p, g in pairs if g in keep_g)

        gt_sub["items"].append({**g_item, "annotations": [G[i] for i in sorted(keep_g)]})
        pred_sub["items"].append({**p_item, "annotations": [P[i] for i in keep_p]})

    return pred_sub, gt_sub, n_unknown


def evaluate_damage_subsets(pred_data, gt_data, iou_threshold):
    """Print the BBCH metrics for plants with and without visible damage.
 
    For each subset, a table with TP, FP, FN, precision, recall, F1, MAE and BIAS
    per BBCH class is printed, followed by the macro and micro scores. BBCH 17 is
    not shown in the table.
    """
    for title, damaged in (("DAMAGE VISIBLE", True), ("NO DAMAGE VISIBLE", False)):
        pred_sub, gt_sub, n_unknown = split_by_damage(pred_data, gt_data, damaged, iou_threshold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            (class_stats, macro_f1, weighted_f1, micro_f1, macro_mae, weighted_mae, micro_mae,
             macro_bias, weighted_bias, micro_bias) = InstanceDetectionMetrics(iou_threshold=iou_threshold).evaluate(pred_sub, gt_sub)
        loc = LocalizationMetrics(iou_threshold=iou_threshold).evaluate(pred_sub, gt_sub)
        n_gt = sum(len(_masks(it)) for it in gt_sub["items"])

        print("\n" + "=" * 105)
        print(f" BBCH EVALUATION - {title} (IoU-Threshold: {iou_threshold}) | "
              f"GT plants: {n_gt} | matched: {loc['tp']} | missed: {loc['fn']}")
        print("=" * 105)
        print(f"{'Class':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<10} | {'Recall':<10} | "
              f"{'F1-Score':<10} | {'MAE':<10} | {'BIAS':<10}")
        print("-" * 105)
        for stats in class_stats.values():
            if stats["name"] == "17":
                continue
            print(
                f"BBCH-{stats['name']:<5} | "
                f"{stats['tp']:<5} | "
                f"{stats['fp']:<5} | "
                f"{stats['fn']:<5} | "
                f"{stats['precision']:<10.4f} | "
                f"{stats['recall']:<10.4f} | "
                f"{stats['f1']:<10.4f} | "
                f"{stats['MAE']:<10.4f} | "
                f"{stats['BIAS']:<10.4f}"
            )
        print("-" * 105)
        print(f"MACRO F1-SCORE: {macro_f1:.4f}")
        print(f"MICRO F1-SCORE: {micro_f1:.4f}")
        print(f"MACRO MAE: {macro_mae:.4f}")
        print(f"MICRO MAE: {micro_mae:.4f}")
        print(f"MACRO BIAS: {macro_bias:.4f}")
        print(f"MICRO BIAS: {micro_bias:.4f}")
        if n_unknown:
            print(f"[WARN] {n_unknown} GT plants without damage_visible attribute.")