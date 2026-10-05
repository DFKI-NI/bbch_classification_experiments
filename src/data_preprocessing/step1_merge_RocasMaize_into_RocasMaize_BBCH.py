
"""Merge RocasMaize annotations into the three BBCH dataset subsets.
 
This script combines the RocasMaize annotations with the RocasMaize_BBCH subsets (train, val, test).
For each subset it does the following steps:
 
- Merge "weed_sprayed" annotations into "weed".
- Match maize masks between RocasMaize and RocasMaize_BBCH. Unmatched maize masks get bbch = -1.
- Assign leaves to maize plants and save the overlap in percent.
- Assign stem keypoints to maize plants and weeds. If a mask has no keypoint, a
  virtual keypoint is created at its centroid.
- Assign global IDs to maize plants (plant_id), leaves (leaf_id), weeds (weed_id)
  and stem keypoints (stem_id).
- Clean the image-level metadata and add the growth cycle (early/late).
- Update the categories in the JSON header: remove the "weed_sprayed" label and
  set the allowed attributes per label.
- Remove unused attributes to keep the JSON files small.
 
main() returns one dict with the three merged subsets. When the script is run
directly, each subset is saved as its own JSON file.
 
Usage:
    python3 src/data_preprocessing/step1_merge_RocasMaize_into_RocasMaize_BBCH.py
"""


import json
import argparse
from pathlib import Path
import pycocotools.mask as mask_util
import numpy as np
from typing import Dict, Any, List, Tuple, Union, Optional
from tqdm import tqdm


SEARCH_RADIUS = 5  # Pixel search radius to match key points and maize masks

OVERLAP = 0.85  # 85% overlap threshold between maize masks of both datasets

CYCLE_THRESHOLDS = {
    1: "2023-08-07T23:59:59",
    2: "2023-07-24T23:59:59",
    3: "2023-07-12T23:59:59",
    4: "2023-08-17T23:59:59",
}

MAIZE_BASE_ATTRIBUTES = [
    "damage_visible",
    "bbch",
    "leaf_count",  
]

IMAGE_LEVEL_ATTRIBUTES = [
    "rope_position_odom_[mm]", 
    "timestamp_iso", 
    "field_number", 
    "row_number",
    "cycle"
]


def get_bbox_from_mask(
    mask: np.ndarray
) -> Tuple[int, int, int, int]:
    """Compute the bounding box of a binary mask.
 
    Returns (x_min, y_min, width, height). Returns all zeros if the mask is empty.
    """
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0, 0, 0, 0
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


def remove_unused_attributes(
    ann: Dict[str, Any],
    keep_attributes: Optional[List[str]] = None
) -> Dict[str, Any]: 
    """Remove all attributes of an annotation except the allowed ones.
 
    If keep_attributes is None, all attributes are removed. The annotation is
    changed in place.

    Returns the cleaned annotation.
    """
    current_attrs = ann.get("attributes", {})
    cleaned_attrs = {}
    if keep_attributes is not None:
        for attr in keep_attributes:
            if attr in current_attrs:
                cleaned_attrs[attr] = current_attrs[attr]
    ann["attributes"] = cleaned_attrs
    return ann


def get_centroid(
    mask: np.ndarray
) -> Tuple[float, float]:
    """Compute the centroid (x, y) of a binary mask.
 
    Returns the centroid (x, y), or (0.0, 0.0) if the mask is empty.
    """
    y_idx, x_idx = np.where(mask > 0)
    if len(y_idx) == 0:
        return 0.0, 0.0
    return float(np.mean(x_idx)), float(np.mean(y_idx))


def extract_image_attributes(
    img_attr: Dict[str, Any]
) -> Dict[str, Any]:
    """Keep only the needed image-level attributes and add the cycle.
 
    The cycle is "early" if the image was taken on or before the threshold date of
    its field, otherwise "late".
    
    Returns the cleaned image attributes.
    """
    cleaned_img_attr = {}

    for key, value in img_attr.items():
        if key in IMAGE_LEVEL_ATTRIBUTES:
            cleaned_img_attr[key] = value

    f_id = img_attr.get("field_number")
    t_iso = img_attr.get("timestamp_iso")

    if f_id is not None and t_iso is not None:
        threshold = CYCLE_THRESHOLDS.get(int(f_id))
        if threshold:
            cleaned_img_attr["cycle"] = "early" if t_iso <= threshold else "late"

    return cleaned_img_attr



def match_maize_annotations(
    rocas_maize_annotations: List[Dict[str, Any]],
    bbch_maize_annotations: List[Dict[str, Any]],
    plant_global_counter: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Match RocasMaize maize masks to RocasMaize_BBCH maize masks and copy the BBCH value.
 
    A RocasMaize mask matches a RocasMaize_BBCH mask if at least OVERLAP (85 %) of its pixels are
    covered by the BBCH mask. Each BBCH mask can be used only once. Matched masks
    get a new plant_id. Unmatched or empty masks get bbch = -1 and plant_id = -1.
 
    Returns the matched masks, the unmatched masks and the updated plant counter.
    """
    matched_bbch_ids = set()
    valid_maize_annotations = []
    unmatched_maize_annotations = []

    for r_ann in rocas_maize_annotations:
        r_ann = remove_unused_attributes(r_ann, keep_attributes=MAIZE_BASE_ATTRIBUTES)
        r_mask = mask_util.decode(r_ann["rle"])
        r_pixel_count = np.sum(r_mask)
        
        if r_pixel_count == 0:
            r_ann["attributes"]["bbch"] = -1
            r_ann["attributes"].update({"plant_id": -1, "matched_leaves": [], "stem_point_id": None})
            unmatched_maize_annotations.append(r_ann)
            continue

        best_match_bbch_val = None
        highest_overlap = 0
        best_match_ann = None

        for b_ann in bbch_maize_annotations:
            if id(b_ann) in matched_bbch_ids:
                continue

            b_mask = mask_util.decode(b_ann["rle"])
            rx, ry, rw, rh = get_bbox_from_mask(r_mask)
            bx, by, bw, bh = get_bbox_from_mask(b_mask)
            if (rx + rw < bx or bx + bw < rx or ry + rh < by or by + bh < ry):
                continue

            intersection = np.logical_and(r_mask, b_mask).sum()
            overlap = intersection / r_pixel_count

            if overlap >= OVERLAP and overlap > highest_overlap:
                highest_overlap = overlap
                best_match_bbch_val = b_ann.get("attributes", {}).get("bbch")
                best_match_ann = b_ann

        if best_match_bbch_val is not None:
            r_ann["attributes"]["bbch"] = best_match_bbch_val
            r_ann["attributes"].update({
                "plant_id": plant_global_counter, 
                "matched_leaves": [], 
                "stem_point_id": None
            })
            plant_global_counter += 1
            if best_match_ann is not None:
                matched_bbch_ids.add(id(best_match_ann))
            valid_maize_annotations.append(r_ann)
        else:
            # unmatched maize (e.g. BBCH 9/18) is marked by bbch = -1
            r_ann["attributes"]["bbch"] = -1
            r_ann["attributes"].update({
                "plant_id": -1, 
                "matched_leaves": [], 
                "stem_point_id": None
            })
            unmatched_maize_annotations.append(r_ann)

    return valid_maize_annotations, unmatched_maize_annotations, plant_global_counter



def associate_leaves_with_plants(
    leaf_annotations: List[Dict[str, Any]],
    valid_maize_annotations: List[Dict[str, Any]],
    all_candidate_plants: List[Dict[str, Any]]
) -> None:
    """Assign each leaf to the maize plant it overlaps the most.
 
    The overlap in percent is saved in the leaf. If the best plant is a valid
    (matched) plant, the leaf gets its plant_id and the plant adds the leaf_id
    to its matched_leaves. Otherwise matched_plant is -1. All annotations are
    changed in place.
    """
    for leaf in leaf_annotations:
        best_plant = None
        max_overlap_pixels = 0
        l_mask = mask_util.decode(leaf["rle"])
        leaf_total_pixels = l_mask.sum()

        if leaf_total_pixels == 0:
            continue

        for plant in all_candidate_plants:
            p_mask = mask_util.decode(plant["rle"])
            overlap = np.logical_and(l_mask, p_mask).sum()

            if overlap > max_overlap_pixels:
                max_overlap_pixels = overlap
                best_plant = plant

        if best_plant is not None and max_overlap_pixels > 0:
            overlap_pct = round((max_overlap_pixels / leaf_total_pixels) * 100, 2)
            leaf["attributes"]["overlap_with_plant"] = overlap_pct

            if best_plant in valid_maize_annotations:
                plant_id = best_plant["attributes"]["plant_id"]
                leaf["attributes"]["matched_plant"] = plant_id
                
                if leaf["attributes"]["leaf_id"] not in best_plant["attributes"]["matched_leaves"]:
                    best_plant["attributes"]["matched_leaves"].append(leaf["attributes"]["leaf_id"])
            else:
                leaf["attributes"]["matched_plant"] = -1
        else:
            leaf["attributes"]["matched_plant"] = -1



def process_stem_keypoints(
    all_masks: List[Dict[str, Any]],
    kp_annotations: List[Dict[str, Any]],
    id_mapping: Dict[int, int],
    stem_global_counter: int
) -> Tuple[List[Dict[str, Any]], int]:
    """Link stem keypoints to maize and weed masks.
 
    For each mask, search for a stem keypoint of the matching class within
    SEARCH_RADIUS pixels of the mask. If one is found, it gets a stem_id (if it has
    none yet) and the mask stores this id as stem_point_id. If none is found, a
    virtual keypoint is created at the mask centroid.
 
    Returns the new virtual keypoints and the updated stem counter.
    """
    virtual_keypoints_to_add = []

    for mask_ann in all_masks:
        lbl_id = mask_ann.get("label_id")
        if lbl_id not in id_mapping:
            continue
            
        target_stem_id = id_mapping[lbl_id]
        m_arr = mask_util.decode(mask_ann["rle"])
        img_h, img_w = m_arr.shape
        
        best_kp = None
        best_dist = float("inf")

        # search for existing key point near the mask
        for kp in kp_annotations:
            if kp.get("label_id") != target_stem_id:
                continue
            
            kx, ky = kp["points"][0], kp["points"][1]
            center_x, center_y = int(round(kx)), int(round(ky))
            
            match_found = False
            for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
                for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
                    cx, cy = center_x + dx, center_y + dy
                    if 0 <= cy < img_h and 0 <= cx < img_w:
                        if m_arr[cy, cx] > 0:
                            d = np.sqrt(dx**2 + dy**2)
                            if d < best_dist:
                                best_dist = d
                                best_kp = kp
                            match_found = True
                            break
                if match_found:
                    break

        if best_kp:  # real key point was found
            if "attributes" not in best_kp:
                best_kp["attributes"] = {}
            # if the real key point does not have an id yet, assign the next free id
            if "stem_id" not in best_kp["attributes"]:
                best_kp["attributes"]["stem_id"] = stem_global_counter
                best_kp["attributes"]["is_virtual"] = False
                current_stem_id = stem_global_counter
                stem_global_counter += 1
            else:
                current_stem_id = best_kp["attributes"]["stem_id"]
            mask_ann["attributes"]["stem_point_id"] = current_stem_id

        else:  # no key point found, create a virtual key point
            cx, cy = get_centroid(m_arr)
            if cx != 0.0 or cy != 0.0:
                current_stem_id = stem_global_counter
                mask_ann["attributes"]["stem_point_id"] = current_stem_id
                new_virtual_kp = {
                    "id": 0,
                    "type": "points",
                    "attributes": {
                        "stem_id": current_stem_id,
                        "is_virtual": True
                    },
                    "label_id": target_stem_id,
                    "points": [float(cx), float(cy)]
                }
                virtual_keypoints_to_add.append(new_virtual_kp)
                stem_global_counter += 1

    return virtual_keypoints_to_add, stem_global_counter



def update_category_schema(
    rocas_data: Dict[str, Any], 
    output_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build the output dataset with an updated category schema.
 
    Makes a deep copy of the RocasMaize dataset, replaces its items with the merged items,
    removes the weed_sprayed label and sets the allowed attributes for each label.

    Returns the output dataset as a new dict.
    """
    output_subset_data = json.loads(json.dumps(rocas_data))
    output_subset_data["items"] = output_items
    
    if "categories" in output_subset_data and "label" in output_subset_data["categories"]:
        label_cfg = output_subset_data["categories"]["label"]
        label_cfg.pop("label_groups", None)
        label_cfg["attributes"] = []
        
        labels = [lab for lab in label_cfg.get("labels", []) if lab.get("name") != "weed_sprayed"]
        label_cfg["labels"] = labels

        for label_obj in labels:
            lname = label_obj.get("name")
            if lname == "maize":
                label_obj["attributes"] = sorted(MAIZE_BASE_ATTRIBUTES + ["plant_id", "matched_leaves", "stem_point_id"])
            elif lname == "maize_leaf":
                label_obj["attributes"] = ["leaf_id", "matched_plant"]
            elif lname in ["maize_stem", "weed_stem"]:
                label_obj["attributes"] = ["stem_id", "is_virtual"]
            elif lname == "weed":
                label_obj["attributes"] = sorted(["weed_id", "stem_point_id"])
            else:
                label_obj["attributes"] = []

    return output_subset_data



def process_subset(
    rocas_data: Dict[str, Any],
    subset_data: Dict[str, Any], 
    subset_name: str
) -> Dict[str, Any]:
    """Merge RocasMaize annotations into one RocasMaize_BBCH subset (train, val or test).
 
    For each BBCH image, the matching RocasMaize image is looked up (by exact id or by
    id suffix). Images without a match are skipped. Then weeds, leaves,
    maize plants and stem keypoints are processed and get global IDs that are
    unique within the subset.
 
    Returns the merged subset as a dataset dict.
    """
    rocas_items_dict = {item["id"]: item for item in rocas_data.get("items", [])}
    rocas_ids = list(rocas_items_dict.keys())
    output_items = []

    label_cfg = rocas_data["categories"]["label"]
    name_to_id = {lbl["name"]: idx for idx, lbl in enumerate(label_cfg.get("labels", []))}

    id_mapping = {}
    weed_label_id = name_to_id.get("weed")
    weed_sprayed_label_id = name_to_id.get("weed_sprayed")
    weed_stem_id = name_to_id.get("weed_stem")
    maize_id = name_to_id.get("maize")
    maize_stem_id = name_to_id.get("maize_stem")

    if maize_id is not None and maize_stem_id is not None:
        id_mapping[maize_id] = maize_stem_id
    if weed_label_id is not None and weed_stem_id is not None:
        id_mapping[weed_label_id] = weed_stem_id

    # counter for ids
    plant_global_counter = 0  # Attribute name = plant_id
    leaf_global_counter = 0  # Attribute name = leaf_id
    stem_global_counter = 0  # Attribute name = stem_id
    weed_global_counter = 0  # Attribute name = weed_id

    subset_items = subset_data.get("items", [])

    for bbch_item in tqdm(subset_items, desc=f"Processing {subset_name}"):
        bbch_id = bbch_item["id"]
        
        if bbch_id in rocas_items_dict:
            rocas_item = rocas_items_dict[bbch_id]
        else:
            matched_id = next((r_id for r_id in rocas_ids if r_id.endswith(bbch_id) or bbch_id.endswith(r_id)), None)
            if matched_id:
                rocas_item = rocas_items_dict[matched_id]
            else:
                continue

        cleaned_img_attr = extract_image_attributes(rocas_item.get("attr", {}))
        rocas_annotations = rocas_item.get("annotations", [])

        # merge weed_sprayed and weed and assign weed_id
        weed_annotations = []
        for a in rocas_annotations:
            if a.get("type") == "mask" and a.get("label_id") not in [0, 1]:
                if weed_sprayed_label_id is not None and a.get("label_id") == weed_sprayed_label_id:
                    a["label_id"] = weed_label_id
                
                a = remove_unused_attributes(a)
                if "attributes" not in a:
                    a["attributes"] = {}

                a["attributes"]["weed_id"] = weed_global_counter
                a["attributes"]["stem_point_id"] = None
                weed_global_counter += 1

                weed_annotations.append(a)

        # extract key point annotations
        kp_annotations = [
            remove_unused_attributes(a) for a in rocas_annotations 
            if a.get("type") == "points" and not a.get("attributes", {}).get("is_virtual", False)
        ]
        
        # prepare maize leaves and assign global leaf_id
        leaf_annotations = [remove_unused_attributes(a) for a in rocas_annotations if a.get("type") == "mask" and a.get("label_id") == 1]
        for l_ann in leaf_annotations:
            l_ann["attributes"] = {
                "leaf_id": leaf_global_counter, 
                "matched_plant": -1, 
                "overlap_with_plant": -1,
            }
            leaf_global_counter += 1

        # Match maize masks between RocasMaize and RocasMaize_BBCH dataset
        rocas_maize_annotations = [a for a in rocas_annotations if a.get("type") == "mask" and a.get("label_id") == 0]
        bbch_maize_annotations = [a for a in bbch_item.get("annotations", []) if a.get("type") == "mask" and a.get("label_id") == 0]
        valid_maize_annotations, filtered_out_maize_annotations, plant_global_counter = match_maize_annotations(
            rocas_maize_annotations, bbch_maize_annotations, plant_global_counter
        )
        all_candidate_plants = valid_maize_annotations + filtered_out_maize_annotations

        # assign leaves to plants
        associate_leaves_with_plants(leaf_annotations, valid_maize_annotations, all_candidate_plants)

        all_masks = valid_maize_annotations + filtered_out_maize_annotations + weed_annotations
        virtual_keypoints_to_add, stem_global_counter = process_stem_keypoints(
            all_masks, kp_annotations, id_mapping, stem_global_counter
        )

        # bring all annotations together and reindex them
        all_image_annotations = all_masks + leaf_annotations + kp_annotations + virtual_keypoints_to_add
        for idx, ann in enumerate(all_image_annotations):
            ann["id"] = idx
        new_item = rocas_item.copy()
        new_item["annotations"] = all_image_annotations
        new_item["attr"] = cleaned_img_attr
        output_items.append(new_item)

    return update_category_schema(rocas_data, output_items)




def main(
    rocasmaize_source: Union[Path, Dict[str, Any]], 
    rocasmaize_bbch_subsets_source: Union[Path, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Load the input data and merge RocasMaize into all RocasMaize_BBCH subsets.
 
    Both inputs can be file paths or already loaded dicts. Missing subsets are skipped.
 
    Returns a dict with the merged subsets ("train", "val", "test").
    """
    if isinstance(rocasmaize_source, Path):
        if not rocasmaize_source.exists():
            raise FileNotFoundError(f"RocasMaize JSON not found at: {rocasmaize_source}")
        with open(rocasmaize_source, "r", encoding="utf-8") as f:
            rocasmaize_data = json.load(f)
    else:
        rocasmaize_data = rocasmaize_source

    results = {}

    # Process each subset split (train, val, test)
    for subset_name in ["train", "val", "test"]:
        if isinstance(rocasmaize_bbch_subsets_source, Path):
            subset_file = rocasmaize_bbch_subsets_source / f"{subset_name}.json"
            if not subset_file.exists():
                print(f"Warning: Subset file {subset_file.name} not found. Skipping.")
                continue
            with open(subset_file, "r", encoding="utf-8") as f:
                subset_data = json.load(f)
        else:
            if subset_name not in rocasmaize_bbch_subsets_source:
                continue
            subset_data = rocasmaize_bbch_subsets_source[subset_name]

        final_subset_json = process_subset(rocasmaize_data, subset_data, subset_name)  # type: ignore
        results[subset_name] = final_subset_json

    return results



if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    parser = argparse.ArgumentParser(description="Merge RocasMaize annotations into the 3 RocasMaize_BBCH subsets.")
    parser.add_argument(
        "--RocasMaize-json", type=str, 
        default=str(PROJECT_ROOT / "datasets" / "RocasMaize" / "annotations_datumaro" / "all.json"),
        help="Path to the RocasMaize dataset default.json file"
    )
    parser.add_argument(
        "--RocasMaize_BBCH-dir", type=str, 
        default=str(PROJECT_ROOT / "datasets" / "RocasMaize_BBCH" / "annotations_datumaro"),
        help="Directory containing the 3 RocasMaize_BBCH subset files"
    )
    parser.add_argument(
        "--output-dir", type=str, 
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step1"),
        help="Directory where the generated subset JSONs will be saved"
    )
    args = parser.parse_args()
    rocasmaize_json = Path(args.RocasMaize_json)
    rocasmaize_bbch_dir = Path(args.RocasMaize_BBCH_dir)
    out_dir = Path(args.output_dir)

    # Execute processing pipeline and export outputs to JSON files
    processed_subsets = main(rocasmaize_source=rocasmaize_json, rocasmaize_bbch_subsets_source=rocasmaize_bbch_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    for subset_name, subset_data in processed_subsets.items():
        out_path = out_dir / f"{subset_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {subset_name}.json to {out_path}")