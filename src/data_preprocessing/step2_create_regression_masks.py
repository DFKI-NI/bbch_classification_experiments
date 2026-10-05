"""Create distance and direction regression masks from the step1 output.
 
For each image, this script creates two pixel-wise regression targets and saves
them in the image attributes (attr["distance_mask"] and attr["direction_mask"]):
 
- Distance mask (H x W): for matched maize plants with BBCH 10 to 17, the
  distance of each pixel to the plant's stem point, divided by the global maximum
  radius. All other pixels get a fixed value: background = -1.0, weed = -0.66,
  maize_leaf and other maize = -0.33.
- Direction mask (2 x H x W): a unit vector (cos, sin) per pixel that points
  away from the reference point of its mask. Background is [0, 0].
 
The reference point of a maize or weed mask is its stem keypoint. For leaf masks
it is found with the rules in LEAF_REFERENCE_ORDER. Masks are drawn by priority,
so matched maize with BBCH 10 to 17 overwrites leaves and weeds.
 
Both masks are stored as float16, compressed with zlib and encoded as base64.
 
Usage:
    python3 src/data_preprocessing/step2_create_regression_masks.py
"""


import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import zlib
import cv2
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm


# Order in which the rules for the leaf reference point are tried.
# Allowed: "stem_in_leaf", "cluster_centroid", "matched_plant", "border","leaf_centroid"
LEAF_REFERENCE_ORDER = ["stem_in_leaf", "cluster_centroid", "leaf_centroid"]

# A leaf counts as touching the border if it is at most this many pixels away
# from the image edge. Only used by the rule "border".
BORDER_TOUCH_TOLERANCE = 2

# Minimum overlap (attribute `overlap_with_plant`, in percent) to accept the
# stem point of the matched plant. Only used by the rule "matched_plant".
MIN_LEAF_OVERLAP = 50.0


def build_keypoint_map(
    annotations: List[Dict[str, Any]]
) -> Dict[str, Tuple[float, float]]:
    """Map each stem_id to the (x, y) position of its keypoint.
 
    Only point annotations with a stem_id and at least two coordinates are used.
    The keys are the stem_ids as strings.

    Returns a dict that maps each stem_id (as a string) to (x, y).
    """
    kp_map = {}
    for ann in annotations:
        if ann.get("type") == "points":
            stem_id = ann.get("attributes", {}).get("stem_id")
            points = ann.get("points", [])
            if stem_id is not None and len(points) >= 2:
                kp_map[str(stem_id)] = (float(points[0]), float(points[1]))
    return kp_map


def get_target_point(
    ann: Dict[str, Any],
    kp_map: Dict[str, Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    """Return the stem keypoint (x, y) of a mask via its stem_point_id.
 
    Returns the keypoint (x, y), or None if the mask has no stem_point_id or the
    keypoint is not found.
    """
    stem_point_id = ann.get("attributes", {}).get("stem_point_id")
    if stem_point_id is not None and str(stem_point_id) in kp_map:
        return kp_map[str(stem_point_id)]
    return None


def get_label_name(
    label_id: int,
    categories_cfg: Dict[str, Any]
) -> str:
    """Return the label name for a label_id from the category schema.
 
    Returns the label name, or an empty string if the label_id is out of range.
    """
    labels = categories_cfg.get("label", {}).get("labels", [])
    if 0 <= label_id < len(labels):
        return labels[label_id].get("name", "")
    return ""


def build_plant_target_map(
    annotations: List[Dict[str, Any]],
    kp_map: Dict[str, Tuple[float, float]],
    categories_cfg: Dict[str, Any]
) -> Dict[int, Tuple[float, float]]:
    """Map each plant_id to the stem keypoint (x, y) of its maize mask.
 
    Maize masks without a plant_id (plant_id = -1) or without a stem keypoint are
    skipped.

    Returns a dict that maps each plant_id to (x, y).
    """
    plant_target_map: Dict[int, Tuple[float, float]] = {}
    for ann in annotations:
        if ann.get("type") != "mask":
            continue
        if get_label_name(ann.get("label_id", -1), categories_cfg) != "maize":
            continue
        plant_id = ann.get("attributes", {}).get("plant_id", -1)
        if plant_id == -1:
            continue
        target_pt = get_target_point(ann, kp_map)
        if target_pt:
            plant_target_map[plant_id] = target_pt
    return plant_target_map


def collect_maize_stem_points(
    annotations: List[Dict[str, Any]],
    categories_cfg: Dict[str, Any]
) -> List[Tuple[float, float, bool]]:
    """Collect all maize_stem keypoints as (x, y, is_assigned).
 
    is_assigned is True if a mask already points to the keypoint via its
    stem_point_id. Unassigned keypoints usually belong to plants whose maize mask
    was removed from the base dataset.

    Returns a list of (x, y, is_assigned) tuples.
    """
    assigned_ids = set()
    for ann in annotations:
        if ann.get("type") == "mask":
            sid = ann.get("attributes", {}).get("stem_point_id")
            if sid is not None:
                assigned_ids.add(sid)

    stem_points: List[Tuple[float, float, bool]] = []
    for ann in annotations:
        if ann.get("type") != "points":
            continue
        if get_label_name(ann.get("label_id", -1), categories_cfg) != "maize_stem":
            continue
        pts = ann.get("points", [])
        if len(pts) < 2:
            continue
        sid = ann.get("attributes", {}).get("stem_id")
        stem_points.append((float(pts[0]), float(pts[1]), sid in assigned_ids))

    return stem_points


def build_leaf_clusters(
    annotations: List[Dict[str, Any]],
    categories_cfg: Dict[str, Any],
    shape: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], Dict[int, Tuple[float, float]]]:
    """Group the leaf pixels outside all maize masks into connected regions.
 
    Returns the label image of the regions (or None if there are no such pixels)
    and a dict that maps each region label to its centroid (x, y).
    """
    maize_union = np.zeros(shape, dtype=bool)
    leaf_union = np.zeros(shape, dtype=bool)
    has_leaf = False

    for ann in annotations:
        if ann.get("type") != "mask" or not ann.get("rle"):
            continue
        name = get_label_name(ann.get("label_id", -1), categories_cfg)
        if name == "maize":
            maize_union |= mask_util.decode(ann["rle"]).astype(bool)
        elif name == "maize_leaf":
            leaf_union |= mask_util.decode(ann["rle"]).astype(bool)
            has_leaf = True

    if not has_leaf:
        return None, {}

    exclusive = leaf_union & ~maize_union
    if not exclusive.any():
        return None, {}

    num_labels, comp_labels = cv2.connectedComponents(
        exclusive.astype(np.uint8), connectivity=8
    )

    centroids: Dict[int, Tuple[float, float]] = {}
    for c in range(1, num_labels):
        ys, xs = np.where(comp_labels == c)
        if len(xs):
            centroids[c] = (float(xs.mean()), float(ys.mean()))

    return comp_labels, centroids


def project_to_nearest_border(
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    img_h: int,
    img_w: int,
    tolerance: int = BORDER_TOUCH_TOLERANCE
) -> Optional[Tuple[float, float]]:
    """Place a virtual stem point on the image border closest to the mask.
 
    Returns the virtual point (x, y), or None if the mask is more than `tolerance`
    pixels away from all borders.
    """
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())

    distances = {
        "left": x_min,
        "right": (img_w - 1) - x_max,
        "top": y_min,
        "bottom": (img_h - 1) - y_max,
    }
    side = min(distances, key=lambda k: distances[k])
    if distances[side] > tolerance:
        return None

    if side == "left":
        sel = x_indices <= x_min + tolerance
        return (-1.0, float(y_indices[sel].mean()))
    if side == "right":
        sel = x_indices >= x_max - tolerance
        return (float(img_w), float(y_indices[sel].mean()))
    if side == "top":
        sel = y_indices <= y_min + tolerance
        return (float(x_indices[sel].mean()), -1.0)

    sel = y_indices >= y_max - tolerance
    return (float(x_indices[sel].mean()), float(img_h))


def resolve_leaf_target_point(
    ann: Dict[str, Any],
    mask_bool: np.ndarray,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    stem_points: List[Tuple[float, float, bool]],
    plant_target_map: Dict[int, Tuple[float, float]],
    comp_labels: Optional[np.ndarray],
    comp_centroids: Dict[int, Tuple[float, float]],
    img_h: int,
    img_w: int
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Find the reference point of a leaf mask.
 
    The rules in LEAF_REFERENCE_ORDER are tried in order and the first rule that
    gives a point is used:
    - stem_in_leaf: an unassigned maize_stem keypoint that lies inside the leaf.
    - cluster_centroid: the centroid of the leaf cluster that covers most of the leaf.
    - matched_plant: the stem point of the matched plant, if the overlap is at least
      MIN_LEAF_OVERLAP.
    - border: a virtual point on the nearest image border.
    - leaf_centroid: the centroid of the leaf itself.
    If no rule gives a point, the leaf centroid is used. 
    
    Returns (point, rule_name).
    """
    attrs = ann.get("attributes", {})

    for rule in LEAF_REFERENCE_ORDER:

        if rule == "stem_in_leaf":
            for x, y, is_assigned in stem_points:
                if is_assigned:
                    continue
                ix, iy = int(round(x)), int(round(y))
                if 0 <= iy < img_h and 0 <= ix < img_w and mask_bool[iy, ix]:
                    return (x, y), rule

        elif rule == "cluster_centroid":
            if comp_labels is not None:
                labels_under = comp_labels[y_indices, x_indices]
                labels_under = labels_under[labels_under > 0]
                if len(labels_under):
                    dominant = int(np.bincount(labels_under).argmax())
                    if dominant in comp_centroids:
                        return comp_centroids[dominant], rule

        elif rule == "matched_plant":
            matched_plant = attrs.get("matched_plant", -1)
            overlap = attrs.get("overlap_with_plant", -1)
            if matched_plant != -1 and overlap is not None and overlap >= MIN_LEAF_OVERLAP:
                pt = plant_target_map.get(matched_plant)
                if pt:
                    return pt, rule

        elif rule == "border":
            pt = project_to_nearest_border(y_indices, x_indices, img_h, img_w)
            if pt is not None:
                return pt, rule

        elif rule == "leaf_centroid":
            return (float(x_indices.mean()), float(y_indices.mean())), rule

    return (float(x_indices.mean()), float(y_indices.mean())), "leaf_centroid"


def get_annotation_priority(
    ann_obj: Dict[str, Any],
    label_name: str
) -> int:
    """Return the drawing priority of a mask.
 
    Masks with a higher priority are drawn later and overwrite masks with a lower
    priority:
        weed                            -> 0
        other maize                     -> 1
        maize_leaf                      -> 1   (same level, both get -0.33)
        matched maize with BBCH 10-17   -> 2

    Returns the priority (0, 1 or 2).
    """

    if label_name == "maize":
        bbch = ann_obj.get("attributes", {}).get("bbch", -1)
        plant_id = ann_obj.get("attributes", {}).get("plant_id", -1)
        return 2 if (10 <= bbch <= 17 and plant_id != -1) else 1
    elif label_name == "maize_leaf":
        return 1
    elif label_name == "weed":
        return 0
    return 0


def find_global_max_radius(
    input_data: Dict[str, Dict[str, Any]],
    split_names: List[str]
) -> float:
    """Find the largest stem-to-pixel distance of all valid maize plants.
 
    Only matched maize plants (plant_id != -1) with BBCH 10 to 17 are used, because
    only they get real distance values. Leaves are not used, since they are never a
    regression target.

    Returns the maximum distance in pixels, or 1.0 if no valid plant is found.
    """
    max_r_global = 0.0

    for split in split_names:
        if split not in input_data:
            continue

        data = input_data[split]
        categories_cfg = data.get("categories", {})

        for item in data.get("items", []):
            annotations = item.get("annotations", [])
            kp_map = build_keypoint_map(annotations)

            for ann in annotations:
                if ann.get("type") != "mask":
                    continue

                if get_label_name(ann.get("label_id", -1), categories_cfg) != "maize":
                    continue

                bbch = ann.get("attributes", {}).get("bbch", -1)
                plant_id = ann.get("attributes", {}).get("plant_id", -1)
                if not (10 <= bbch <= 17 and plant_id != -1):
                    continue

                target_pt = get_target_point(ann, kp_map)
                rle = ann.get("rle")

                if target_pt and rle:
                    mask = mask_util.decode(rle).astype(bool)
                    if not np.any(mask):
                        continue

                    y_indices, x_indices = np.where(mask)
                    dist_map = np.sqrt((x_indices - target_pt[0])**2 + (y_indices - target_pt[1])**2)
                    current_max = float(np.max(dist_map))

                    if current_max > max_r_global:
                        max_r_global = current_max

    if max_r_global == 0.0:
        return 1.0

    print(f"Global Max-Radius: {max_r_global:.2f} pixels")
    return max_r_global


def encode_mask_to_base64(
    array: np.ndarray
) -> Dict[str, Any]:
    """Convert an array to float16, compress it with zlib and encode it as base64.
 
    Returns a dict with the shape, dtype, encoding and the base64 string.
    """
    array_f16 = array.astype(np.float16)
    raw_bytes = array_f16.tobytes()
    compressed_bytes = zlib.compress(raw_bytes, level=6)
    b64_string = base64.b64encode(compressed_bytes).decode('utf-8')

    return {
        "shape": list(array.shape),
        "dtype": "float16",
        "encoding": "zlib+base64",
        "data": b64_string
    }


def process_splits(
    input_source: Union[Path, Dict[str, Dict[str, Any]]],
    split_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Create the distance and direction masks for all splits.
 
    The input can be a directory with the step1 JSON files or a dict with already
    loaded splits. The masks are added to each image as attr["distance_mask"] and
    attr["direction_mask"].
 
    Returns the splits with the added masks.
    """
    input_data: Dict[str, Dict[str, Any]] = {}
    if isinstance(input_source, Path):
        for split in split_names:
            json_path = input_source / f"{split}.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    input_data[split] = json.load(f)
    else:
        input_data = input_source

    global_max_radius = find_global_max_radius(input_data, split_names)

    leaf_rule_stats: Dict[str, int] = {}

    for split in split_names:
        if split not in input_data:
            continue

        data = input_data[split]
        categories_cfg = data.get("categories", {})

        for item in tqdm(data.get("items", []), desc=f"Generating regression masks for {split}"):
            annotations = item.get("annotations", [])

            img_h = item.get("image", {}).get("height")
            img_w = item.get("image", {}).get("width")

            if not img_h or not img_w:
                for ann in annotations:
                    if ann.get("type") == "mask" and "rle" in ann:
                        img_h, img_w = ann["rle"]["size"]
                        break

            if not img_h or not img_w:
                continue

            dist_target = np.full((img_h, img_w), -1.0, dtype=np.float32)
            dir_target = np.zeros((2, img_h, img_w), dtype=np.float32)

            kp_map = build_keypoint_map(annotations)
            stem_points = collect_maize_stem_points(annotations, categories_cfg)
            plant_target_map = build_plant_target_map(annotations, kp_map, categories_cfg)

            comp_labels, comp_centroids = (None, {})
            if "cluster_centroid" in LEAF_REFERENCE_ORDER:
                comp_labels, comp_centroids = build_leaf_clusters(
                    annotations, categories_cfg, (img_h, img_w)
                )

            mask_annotations = []
            for ann in annotations:
                if ann.get("type") == "mask":
                    lbl_name = get_label_name(ann.get("label_id", -1), categories_cfg)
                    if lbl_name in ["maize", "maize_leaf", "weed"]:
                        prio = get_annotation_priority(ann, lbl_name)
                        mask_annotations.append((prio, lbl_name, ann))

            mask_annotations.sort(key=lambda x: x[0])

            for _, label_name, ann in mask_annotations:
                rle = ann.get("rle")
                if not rle:
                    continue

                mask_bool = mask_util.decode(rle).astype(bool)
                if not np.any(mask_bool):
                    continue

                y_indices, x_indices = np.where(mask_bool)

                if label_name == "maize_leaf":
                    target_pt, rule = resolve_leaf_target_point(
                        ann, mask_bool, y_indices, x_indices,
                        stem_points, plant_target_map,
                        comp_labels, comp_centroids, img_h, img_w
                    )
                    leaf_rule_stats[rule] = leaf_rule_stats.get(rule, 0) + 1
                else:
                    target_pt = get_target_point(ann, kp_map)

                if not target_pt:
                    continue

                dx = x_indices - target_pt[0]
                dy = y_indices - target_pt[1]

                theta = np.arctan2(dy, dx)
                dir_target[0, y_indices, x_indices] = np.cos(theta)
                dir_target[1, y_indices, x_indices] = np.sin(theta)

                if label_name == "maize":
                    bbch = ann.get("attributes", {}).get("bbch", -1)
                    plant_id = ann.get("attributes", {}).get("plant_id", -1)

                    if 10 <= bbch <= 17 and plant_id != -1:
                        r = np.sqrt(dx**2 + dy**2)
                        dist_target[y_indices, x_indices] = r / global_max_radius
                    else:
                        dist_target[y_indices, x_indices] = -0.33

                elif label_name == "maize_leaf":
                    dist_target[y_indices, x_indices] = -0.33

                else:
                    dist_target[y_indices, x_indices] = -0.66

            if "attr" not in item:
                item["attr"] = {}

            item["attr"]["distance_mask"] = encode_mask_to_base64(dist_target)
            item["attr"]["direction_mask"] = encode_mask_to_base64(dir_target)

    return input_data


def main(
    input_source: Union[Path, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Create the regression masks for the train, val and test splits.
 
    Returns the splits with the added masks.
    """
    split_names = ["train", "val", "test"]
    return process_splits(input_source=input_source, split_names=split_names)


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    parser = argparse.ArgumentParser(description="Create regression distance and direction masks as RLEs in image attr.")
    parser.add_argument(
        "--input-dir", type=str,
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step1")
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step2")
    )

    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    results = main(input_source=in_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    for subset_name, subset_data in results.items():
        out_path = out_dir / f"{subset_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {subset_name}.json to {out_path}")