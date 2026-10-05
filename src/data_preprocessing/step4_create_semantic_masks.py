"""Create semantic segmentation masks for the cropped plants from step3.
 
For each cropped plant, this script creates a uint8 mask with four classes and
saves it in the image attributes as attr["semantic_mask"]:
 
    0 = background
    1 = weed
    2 = neighbor (neighbor maize plants and all maize leaves)
    3 = main_plant
 
Masks are drawn by priority, so main_plant overwrites neighbor and neighbor
overwrites weed. Leaf masks always get the class neighbor, also the leaves of the
main plant. Only the main plant mask itself gets the class main_plant.
The mask is stored as uint8, compressed with zlib and encoded as base64.
 
Usage:
    python3 src/data_preprocessing/step4_create_semantic_masks.py
"""


import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import zlib
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm


CLASS_MAP = {
    "background": 0,
    "weed": 1,
    "neighbor": 2,
    "main_plant": 3
}


def encode_mask_to_base64(array: np.ndarray) -> Dict[str, Any]:
    """Convert a mask to uint8, compress it with zlib and encode it as base64.
 
    Unlike COCO RLE, this format can store more than two classes in one mask.
 
    Returns a dict with the shape, dtype, encoding and the base64 string.
    """
    array_u8 = array.astype(np.uint8)
    raw_bytes = array_u8.tobytes()
    compressed_bytes = zlib.compress(raw_bytes, level=6)
    b64_string = base64.b64encode(compressed_bytes).decode("utf-8")

    return {
        "shape": list(array.shape),
        "dtype": "uint8",
        "encoding": "zlib+base64",
        "data": b64_string,
    }


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


def extract_target_plant_id(
    item_id: str
) -> int:
    """Read the target plant_id from a cropped item ID.
 
    The item ID has the format "{img_id}_plant_{plant_id}" (see step3).
 
    Returns the plant_id, or -1 if the ID does not have this format.
    """
    if "_plant_" in item_id:
        try:
            return int(item_id.rsplit("_plant_", 1)[1])
        except ValueError:
            pass
    return -1


def generate_segmentation_mask(
    item: Dict[str, Any],
    categories_cfg: Dict[str, Any]
) -> np.ndarray:
    """Create the semantic mask (see CLASS_MAP) for one cropped item.
 
    A maize mask is the main plant if is_main_plant is True. If the attribute is
    missing, its plant_id is compared with the plant_id from the item ID.
 
    Returns the mask as a uint8 array with the size of the crop.
    """
    img_h = item["image"]["size"][0]
    img_w = item["image"]["size"][1]

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    target_plant_id = extract_target_plant_id(str(item.get("id", "")))

    annotations = item.get("annotations", [])

    def get_target_class_and_priority(ann: Dict[str, Any]) -> Tuple[int, int]:
        """Return the class and the drawing priority of a mask.
 
        Masks with an unknown label get the background class and priority 0.
 
        Returns (class_id, priority).
        """
        label_name = get_label_name(ann.get("label_id", -1), categories_cfg)
        attrs = ann.get("attributes", {})

        if label_name == "weed":
            return CLASS_MAP["weed"], 1

        elif label_name == "maize":
            is_main = attrs.get("is_main_plant", None)
            plant_id = attrs.get("plant_id", -1)

            if is_main is True or (is_main is None and target_plant_id != -1 and plant_id == target_plant_id):
                return CLASS_MAP["main_plant"], 3
            return CLASS_MAP["neighbor"], 2

        elif label_name == "maize_leaf":
            return CLASS_MAP["neighbor"], 2

        return CLASS_MAP["background"], 0

    # Collect mask annotations and determine drawing order (higher priority overwrites lower)
    mask_annotations = []
    for ann in annotations:
        if ann.get("type") == "mask":
            target_class, priority = get_target_class_and_priority(ann)
            if target_class != CLASS_MAP["background"]:
                mask_annotations.append((priority, target_class, ann))

    mask_annotations.sort(key=lambda x: x[0])

    for _, target_class, ann in mask_annotations:
        rle = ann.get("rle")
        if not rle:
            continue

        mask_bool = mask_util.decode(rle).astype(bool)
        full_mask[mask_bool] = target_class

    return full_mask


def process_splits(
    input_source: Union[Path, Dict[str, Dict[str, Any]]],
    split_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Create the semantic masks for all splits.
 
    The input can be a directory with the step3 JSON files or a dict with already
    loaded splits. The mask is added to each item as attr["semantic_mask"].
 
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

    for split in split_names:
        if split not in input_data:
            continue

        data = input_data[split]
        categories_cfg = data.get("categories", {})
        items = data.get("items", [])

        for item in tqdm(items, desc=f"Generating semantic masks for {split}"):
            mask_array = generate_segmentation_mask(item, categories_cfg)

            if "attr" not in item:
                item["attr"] = {}

            # Multi-class mask array encoded as base64+zlib instead of COCO RLE
            item["attr"]["semantic_mask"] = encode_mask_to_base64(mask_array)

    return input_data


def main(
    input_source: Union[Path, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Create the semantic masks for the train, val and test splits.
 
    Returns the splits with the added masks.
    """
    split_names = ["train", "val", "test"]
    return process_splits(
        input_source=input_source,
        split_names=split_names
    )


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    parser = argparse.ArgumentParser(description="Generate multi-class semantic segmentation masks directly in JSON items under attr.semantic_mask.")
    parser.add_argument(
        "--input-dir", type=str,
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step3"),
        help="Directory containing Step 3 outputs"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step4"),
        help="Directory where Step 4 JSON outputs will be saved"
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