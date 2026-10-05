"""Create two Mask2Former datasets from the step1 output.
 
This script reads the merged step1 subsets (train, val, test) and creates two
datasets for Mask2Former:
 
- maize_class: one class "maize". Only maize masks with BBCH 10 to 17 are kept,
  together with their bbch and damage_visible attributes.
- bbch_classes: one class per BBCH stage ("10" to "17", label IDs 0 to 7). Only
  maize masks with BBCH 10 to 17 are kept, together with damage_visible.
 
Images without any valid maize mask are removed. Image-level attributes are
cleared. The results are saved as train.json, val.json and test.json in the
folders mask2former_maize_class and mask2former_bbch_classes.
 
Usage:
    python3 src/data_preprocessing/step2_create_mask2former_datasets.py
"""


import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from tqdm import tqdm


ALLOWED_BBCH_STAGES = list(range(10, 18))


def is_valid_bbch_maize(
    ann: Dict[str, Any],
    maize_label_id: int
) -> Tuple[bool, int]:
    """Check if an annotation is a maize mask with a valid BBCH stage.
 
    A BBCH stage is valid if it is between 10 and 17. Returns (True, bbch) for a
    valid mask and (False, -1) otherwise.
    """
    if ann.get("type") != "mask" or ann.get("label_id") != maize_label_id:
        return False, -1

    bbch_val = ann.get("attributes", {}).get("bbch")
    if bbch_val is None:
        return False, -1

    try:
        bbch_int = int(float(bbch_val))
        if bbch_int in ALLOWED_BBCH_STAGES:
            return True, bbch_int
    except (ValueError, TypeError):
        pass

    return False, -1


def get_maize_label_id(
    data: Dict[str, Any]
) -> int:
    """Return the label ID of the "maize" class.
 
    Returns the label ID, or 0 if no "maize" label is found in the categories.
    """
    labels = data.get("categories", {}).get("label", {}).get("labels", [])
    for idx, lbl in enumerate(labels):
        if lbl.get("name") == "maize":
            return idx
    return 0


def process_maize_class_subset(
    subset_data: Dict[str, Any],
    subset_name: str
) -> Dict[str, Any]:
    """Create the maize_class dataset for one subset.
 
    Keeps only maize masks with a valid BBCH stage and sets their label_id to 0.
    Each kept mask only keeps the bbch and damage_visible attributes. Images without
    valid masks are removed and image-level attributes are cleared.
 
    Returns the filtered subset as a new dataset dict.
    """
    output_data = copy.deepcopy(subset_data)
    maize_label_id = get_maize_label_id(output_data)

    if "categories" in output_data and "label" in output_data["categories"]:
        label_cfg = output_data["categories"]["label"]
        label_cfg["attributes"] = ["bbch", "damage_visible"]
        label_cfg["labels"] = [{"name": "maize", "attributes": ["bbch", "damage_visible"]}]

    new_items = []
    items = output_data.get("items", [])
    for item in tqdm(items, desc=f"[{subset_name}] Filtering maize_class"):
        item["attr"] = {}
        filtered_anns = []
        for ann in item.get("annotations", []):
            is_valid, bbch_int = is_valid_bbch_maize(ann, maize_label_id)
            if is_valid:
                ann["label_id"] = 0
                ann["attributes"] = {"bbch": bbch_int,
                                     "damage_visible": ann.get("attributes", {}).get("damage_visible")}
                filtered_anns.append(ann)

        if filtered_anns:
            for idx, ann in enumerate(filtered_anns):
                ann["id"] = idx
            item["annotations"] = filtered_anns
            new_items.append(item)

    output_data["items"] = new_items
    return output_data


def process_bbch_classes_subset(
    subset_data: Dict[str, Any],
    subset_name: str
) -> Dict[str, Any]:
    """Create the bbch_classes dataset for one subset.
 
    Keeps only maize masks with a valid BBCH stage and maps each stage to its own
    class ("10" to "17", label IDs 0 to 7). Each kept mask only keeps the
    damage_visible attribute. Images without valid masks are removed and image-level
    attributes are cleared.
 
    Returns the remapped subset as a new dataset dict.
    """
    output_data = copy.deepcopy(subset_data)
    maize_label_id = get_maize_label_id(output_data)

    bbch_to_id = {stage: idx for idx, stage in enumerate(ALLOWED_BBCH_STAGES)}

    if "categories" in output_data and "label" in output_data["categories"]:
        label_cfg = output_data["categories"]["label"]
        label_cfg["attributes"] = ["damage_visible"]
        label_cfg["labels"] = [{"name": str(stage), "attributes": ["damage_visible"]} for stage in ALLOWED_BBCH_STAGES]

    new_items = []
    items = output_data.get("items", [])
    for item in tqdm(items, desc=f"[{subset_name}] Remapping bbch_classes"):
        item["attr"] = {}
        remapped_anns = []
        for ann in item.get("annotations", []):
            is_valid, bbch_int = is_valid_bbch_maize(ann, maize_label_id)
            if is_valid:
                ann["label_id"] = bbch_to_id[bbch_int]
                ann["attributes"] = {"damage_visible": ann.get("attributes", {}).get("damage_visible")}
                remapped_anns.append(ann)

        if remapped_anns:
            for idx, ann in enumerate(remapped_anns):
                ann["id"] = idx
            item["annotations"] = remapped_anns
            new_items.append(item)

    output_data["items"] = new_items
    return output_data


def verify_annotation_counts(
    dir_maize: Path,
    dir_bbch: Path
) -> None:
    """Compare the number of images and annotations of both datasets.
 
    Reads the saved train, val and test files of both datasets and prints a table.
    The status is "OK" if both datasets have the same number of images and
    annotations, otherwise "ERROR".
    """
    subsets = ["train", "val", "test"]

    print("\n--- Summary Verification ---")
    print(f"{'Subset':<8} | {'Maize Imgs':<10} | {'BBCH Imgs':<10} | {'Maize Anns':<12} | {'BBCH Anns':<12} | {'Status'}")
    print("-" * 75)

    for subset in subsets:
        path_maize = dir_maize / "annotations" / f"{subset}.json"
        path_bbch = dir_bbch / "annotations" / f"{subset}.json"

        img_maize, img_bbch = 0, 0
        count_maize, count_bbch = 0, 0

        if path_maize.exists():
            with open(path_maize, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data.get("items", [])
                img_maize = len(items)
                count_maize = sum(len(item.get("annotations", [])) for item in items)

        if path_bbch.exists():
            with open(path_bbch, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data.get("items", [])
                img_bbch = len(items)
                count_bbch = sum(len(item.get("annotations", [])) for item in items)

        status = "OK" if (count_maize == count_bbch and img_maize == img_bbch) else "ERROR"
        print(f"{subset:<8} | {img_maize:<10} | {img_bbch:<10} | {count_maize:<12} | {count_bbch:<12} | {status}")


def main(
    input_source: Union[Path, Dict[str, Dict[str, Any]]], 
    output_dir: Path
) -> None:
    """Create both Mask2Former datasets for all subsets and save them.
 
    The input can be a directory with the step1 JSON files or a dict with already
    loaded subsets. Missing subsets are skipped. The results are saved in the
    folders mask2former_maize_class and mask2former_bbch_classes in output_dir.
    """
    dir_maize = output_dir / "mask2former_maize_class"
    dir_bbch = output_dir / "mask2former_bbch_classes"

    ann_dir_maize = dir_maize / "annotations"
    ann_dir_bbch = dir_bbch / "annotations"

    ann_dir_maize.mkdir(parents=True, exist_ok=True)
    ann_dir_bbch.mkdir(parents=True, exist_ok=True)

    subsets = ["train", "val", "test"]

    for subset in subsets:
        if isinstance(input_source, Path):
            subset_file = input_source / f"{subset}.json"
            if not subset_file.exists():
                print(f"Warning: {subset_file} not found. Skipping.")
                continue
            with open(subset_file, "r", encoding="utf-8") as f:
                subset_data = json.load(f)
        else:
            if subset not in input_source:
                continue
            subset_data = input_source[subset]

        print(f"\nProcessing Mask2Former split: {subset}...")

        # Process maize_class dataset
        maize_json = process_maize_class_subset(subset_data, subset)
        out_maize_path = ann_dir_maize / f"{subset}.json"
        with open(out_maize_path, "w", encoding="utf-8") as f:
            json.dump(maize_json, f, indent=4, ensure_ascii=False)

        # Process bbch_classes dataset
        bbch_json = process_bbch_classes_subset(subset_data, subset)
        out_bbch_path = ann_dir_bbch / f"{subset}.json"
        with open(out_bbch_path, "w", encoding="utf-8") as f:
            json.dump(bbch_json, f, indent=4, ensure_ascii=False)

    # verify_annotation_counts(dir_maize, dir_bbch)


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    parser = argparse.ArgumentParser(
        description="Extract and remap step1 datasets for Mask2Former (maize vs BBCH classes)."
    )
    parser.add_argument(
        "--input-dir", type=str,
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step1"),
        help="Input directory containing step1 merged JSON subset files (train.json, val.json, test.json)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step2"),
        help="Output base directory to create subfolders for mask2former datasets"
    )

    args = parser.parse_args()
    main(input_source=Path(args.input_dir), output_dir=Path(args.output_dir))