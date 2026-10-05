"""Keep only the main maize plant annotation of each cropped item.
 
This script reads the step4 output and removes all annotations from each cropped
item except the main maize plant (is_main_plant = True). Leaves, weeds, keypoints
and neighbor plants are removed. Only the "maize" label is kept in the category
schema. The masks in the image attributes (attr) are not changed.
 
Usage:
    python3 src/data_preprocessing/step5_filter_target_labels.py
"""
 

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Union
from tqdm import tqdm


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


def filter_subset_annotations(
    subset_data: Dict[str, Any], 
    target_label_id: int = 0,
    subset_name: str = ""
) -> Dict[str, Any]:
    """Remove all annotations except the main maize plant from each item.
 
    An annotation is kept if it has the label "maize" (or target_label_id) and
    is_main_plant is True. The flag is read from the annotation itself or from its
    attributes. The subset is changed in place.
 
    Returns the filtered subset.
    """
    items = subset_data.get("items", [])
    categories_cfg = subset_data.get("categories", {})

    total_kept = 0
    total_removed_secondary = 0
    total_removed_other_class = 0

    for item in tqdm(items, desc=f"Filtering {subset_name}", unit="item"):
        annotations = item.get("annotations", [])
        filtered_annotations = []

        for ann in annotations:
            label_id = ann.get("label_id")
            label_name = get_label_name(label_id, categories_cfg)
            
            if label_id == target_label_id or label_name == "maize":
                attrs = ann.get("attributes", {})
                
                is_main = ann.get("is_main_plant")
                if is_main is None:
                    is_main = attrs.get("is_main_plant", False)
                
                if is_main is True:
                    filtered_annotations.append(ann)
                    total_kept += 1
                else:
                    total_removed_secondary += 1
            else:
                total_removed_other_class += 1

        item["annotations"] = filtered_annotations

    if "categories" in subset_data and "label" in subset_data["categories"]:
        labels = subset_data["categories"]["label"].get("labels", [])
        subset_data["categories"]["label"]["labels"] = [
            lbl for lbl in labels if lbl.get("name") == "maize"
        ]

    return subset_data


def main(
    input_source: Union[Path, Dict[str, Dict[str, Any]]], 
    target_label_id: int = 0
) -> Dict[str, Dict[str, Any]]:
    """Filter the annotations of the train, val and test subsets.
 
    The input can be a directory with the step4 JSON files or a dict with already
    loaded subsets. Missing subsets are skipped.
 
    Returns a dict with the filtered subsets.
    """
    results = {}

    for subset_name in ["train", "val", "test"]:
        if isinstance(input_source, Path):
            subset_file = input_source / f"{subset_name}.json"
            if not subset_file.exists():
                print(f"Warning: Subset file {subset_file.name} not found in input directory. Skipping.")
                continue
            with open(subset_file, "r", encoding="utf-8") as f:
                subset_data = json.load(f)
        else:
            if subset_name not in input_source:
                continue
            subset_data = input_source[subset_name]

        processed_data = filter_subset_annotations(
            subset_data, 
            target_label_id=target_label_id, 
            subset_name=subset_name
        )
        results[subset_name] = processed_data

    return results


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    parser = argparse.ArgumentParser(description="Filter dataset annotations to retain only the main target maize plant per crop.")
    parser.add_argument(
        "--input-dir", type=str, 
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step4"),
        help="Directory containing output JSONs from step 04"
    )
    parser.add_argument(
        "--output-dir", type=str, 
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step5"),
        help="Directory where filtered subset JSONs will be saved"
    )
    parser.add_argument(
        "--target-label-id", type=int, default=0,
        help="Label ID to retain (default: 0 for maize)"
    )

    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    processed_subsets = main(input_source=in_dir, target_label_id=args.target_label_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    for subset_name, subset_data in processed_subsets.items():
        out_path = out_dir / f"{subset_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {subset_name}.json to {out_path}")