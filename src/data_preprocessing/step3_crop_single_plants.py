"""Crop single target maize plants from the full images.
 
This script reads the step2 output and creates one new dataset item per target
maize plant. A target plant is a matched maize plant (plant_id != -1) with BBCH
10 to 17 and a stem point (stem_point_id is set).
 
For each target plant it does the following steps:
1. Compute the bounding box of the plant mask only (no padding).
2. Crop the original image and save it in crop_images_output_dir/{split}/.
3. Crop the regression masks (distance_mask and direction_mask).
4. Crop all annotations inside the box (target plant, leaves, weeds and
   neighbor plants). Keypoints are moved to the crop coordinates.
5. Set is_main_plant = True for the target plant and False for all neighbor
   maize plants.
6. Set the distance mask to -0.33 for all neighbor maize plants.
7. Save the crop as its own dataset item in the output JSON.
 
Notes:
- The crop box only uses the plant mask, not the matched leaves.
  Note that the crop size also affects global_scale in the training
  and therefore the GSD scaling factor.
- Leaf masks are not changed in the distance mask. Only neighbor maize plants
  are set to -0.33.
 
Usage:
    python3 src/data_preprocessing/step3_crop_single_plants.py
"""


import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
import zlib
import cv2
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm


def decode_mask_from_base64(
    mask_dict: Dict[str, Any]
) -> np.ndarray:
    """Decode a mask dict that was created by encode_mask_to_base64.
 
    Returns the mask as a float32 array.
    """
    shape = mask_dict["shape"]
    b64_string = mask_dict["data"]

    compressed_bytes = base64.b64decode(b64_string)
    raw_bytes = zlib.decompress(compressed_bytes)

    array = np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape)
    return array.astype(np.float32)


def encode_mask_to_base64(
    array: np.ndarray
) -> Dict[str, Any]:
    """Convert an array to float16, compress it with zlib and encode it as base64.
 
    Returns a dict with the shape, dtype, encoding and the base64 string.
    """
    array_f16 = array.astype(np.float16)
    raw_bytes = array_f16.tobytes()
    compressed_bytes = zlib.compress(raw_bytes, level=6)
    b64_string = base64.b64encode(compressed_bytes).decode("utf-8")

    return {
        "shape": list(array.shape),
        "dtype": "float16",
        "encoding": "zlib+base64",
        "data": b64_string,
    }


def get_bbox_from_mask(
    mask: np.ndarray
) -> Tuple[int, int, int, int]:
    """Compute the bounding box of a binary mask.
 
    Unlike in step1, the box is given as corner coordinates and not as width and
    height. x_max and y_max are exclusive.
 
    Returns (x_min, y_min, x_max, y_max), or all zeros if the mask is empty.
    """
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0, 0, 0, 0
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    return x_min, y_min, x_max + 1, y_max + 1


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


def process_splits(
    input_source: Union[Path, Dict[str, Dict[str, Any]]],
    images_dir: Path,
    crop_images_output_dir: Path,
    split_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Crop all target maize plants of all splits.
 
    The input can be a directory with the step2 JSON files or a dict with already
    loaded splits. The cropped images are saved in crop_images_output_dir/{split}/.
    Images whose file is not found in images_dir are skipped. The "maize" label gets
    the new attribute is_main_plant.
 
    Returns the splits with one cropped item per target plant.
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

    output_results: Dict[str, Dict[str, Any]] = {}

    for split in split_names:
        if split not in input_data:
            continue

        data = input_data[split]
        categories_cfg = data.get("categories", {})
        split_crop_img_dir = crop_images_output_dir / split
        split_crop_img_dir.mkdir(parents=True, exist_ok=True)

        cropped_items = []

        for item in tqdm(data.get("items", []), desc=f"Cropping plants in {split}"):
            img_id = item["id"]

            img_path = images_dir / f"{img_id}.png"
            if not img_path.exists():
                img_path = images_dir / f"{img_id}.jpg"
                if not img_path.exists():
                    print(f"Warning: Image file for {img_id} not found in {images_dir}. Skipping.")
                    continue

            image_np = cv2.imread(str(img_path))
            if image_np is None:
                continue

            img_h, img_w = image_np.shape[:2]

            dist_mask_full = None
            dir_mask_full = None
            if "attr" in item and "distance_mask" in item["attr"]:
                dist_mask_full = decode_mask_from_base64(item["attr"]["distance_mask"])
            if "attr" in item and "direction_mask" in item["attr"]:
                dir_mask_full = decode_mask_from_base64(item["attr"]["direction_mask"])

            annotations = item.get("annotations", [])

            # Extract target maize plant masks (BBCH 10 to 17, valid plant_id, stem point)
            plant_anns = []
            for ann in annotations:
                if ann.get("type") == "mask":
                    lbl_name = get_label_name(ann.get("label_id", -1), categories_cfg)
                    if lbl_name == "maize":
                        attrs = ann.get("attributes", {})
                        bbch = attrs.get("bbch", -1)
                        plant_id = attrs.get("plant_id", -1)
                        stem_point_id = attrs.get("stem_point_id")
                        if 10 <= bbch <= 17 and plant_id != -1 and stem_point_id is not None:
                            plant_anns.append(ann)

            for plant_ann in plant_anns:
                plant_id = plant_ann["attributes"]["plant_id"]
                p_mask = mask_util.decode(plant_ann["rle"]).astype(bool)

                x_min, y_min, x_max, y_max = get_bbox_from_mask(p_mask)
                if x_max == 0 or y_max == 0:
                    continue

                crop_x1 = max(0, x_min)
                crop_y1 = max(0, y_min)
                crop_x2 = min(img_w, x_max)
                crop_y2 = min(img_h, y_max)

                crop_h = crop_y2 - crop_y1
                crop_w = crop_x2 - crop_x1

                cropped_img = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_img_filename = f"{img_id}_plant_{plant_id}.png"
                cv2.imwrite(str(split_crop_img_dir / crop_img_filename), cropped_img)

                dist_crop = dist_mask_full[crop_y1:crop_y2, crop_x1:crop_x2].copy() if dist_mask_full is not None else None
                dir_crop = dir_mask_full[:, crop_y1:crop_y2, crop_x1:crop_x2].copy() if dir_mask_full is not None else None

                # Crop and Adjust ALL Object Annotations overlapping with the bounding box
                cropped_annotations = []
                new_ann_id = 0

                for ann in annotations:
                    ann_type = ann.get("type")
                    if ann_type == "mask":
                        mask = mask_util.decode(ann["rle"])
                        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

                        if np.any(crop_mask):
                            rle_crop = mask_util.encode(np.asfortranarray(crop_mask))
                            rle_crop["counts"] = rle_crop["counts"].decode("utf-8")  # type: ignore

                            new_ann = ann.copy()
                            new_ann["id"] = new_ann_id
                            new_ann["rle"] = rle_crop

                            lbl_name = get_label_name(ann.get("label_id", -1), categories_cfg)

                            if lbl_name == "maize":
                                if "attributes" not in new_ann:
                                    new_ann["attributes"] = {}
                                else:
                                    new_ann["attributes"] = new_ann["attributes"].copy()

                                ann_plant_id = new_ann["attributes"].get("plant_id", -1)
                                is_main = (ann_plant_id != -1) and (ann_plant_id == plant_id)
                                new_ann["attributes"]["is_main_plant"] = is_main

                                if not is_main and dist_crop is not None:
                                    dist_crop[crop_mask > 0] = -0.33

                            cropped_annotations.append(new_ann)
                            new_ann_id += 1

                    elif ann_type == "points":
                        px, py = ann["points"][0], ann["points"][1]
                        if crop_x1 <= px < crop_x2 and crop_y1 <= py < crop_y2:
                            new_ann = ann.copy()
                            new_ann["id"] = new_ann_id
                            new_ann["points"] = [float(px - crop_x1), float(py - crop_y1)]
                            cropped_annotations.append(new_ann)
                            new_ann_id += 1

                # Re-encode updated regression masks back into image attributes
                cropped_attr = item.get("attr", {}).copy()
                if dist_crop is not None:
                    cropped_attr["distance_mask"] = encode_mask_to_base64(dist_crop)
                if dir_crop is not None:
                    cropped_attr["direction_mask"] = encode_mask_to_base64(dir_crop)

                # Construct new cropped dataset item
                cropped_item = {
                    "id": f"{img_id}_plant_{plant_id}",
                    "image": {
                        "file_name": crop_img_filename,
                        "path": crop_img_filename,
                        "size": [crop_h, crop_w]  # height, width
                    },
                    "annotations": cropped_annotations,
                    "attr": cropped_attr
                }
                cropped_items.append(cropped_item)

        # Save structured JSON output per subset and update categories schema
        out_subset = json.loads(json.dumps(data))
        out_subset["items"] = cropped_items

        if "categories" in out_subset and "label" in out_subset["categories"]:
            for label_obj in out_subset["categories"]["label"].get("labels", []):
                if label_obj.get("name") == "maize":
                    if "attributes" in label_obj and "is_main_plant" not in label_obj["attributes"]:
                        label_obj["attributes"].append("is_main_plant")
                        label_obj["attributes"] = sorted(label_obj["attributes"])

        output_results[split] = out_subset

    return output_results


def main(
    input_source: Union[Path, Dict[str, Dict[str, Any]]],
    images_dir: Path,
    crop_images_output_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """Crop the target maize plants of the train, val and test splits.
 
    Returns the splits with one cropped item per target plant.
    """
    split_names = ["train", "val", "test"]
    return process_splits(
        input_source=input_source,
        images_dir=images_dir,
        crop_images_output_dir=crop_images_output_dir,
        split_names=split_names,
    )


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    parser = argparse.ArgumentParser(description="Crop single maize plants to exact minimal bounding box and update annotation data.")
    parser.add_argument(
        "--input-dir", type=str,
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step2"),
        help="Directory containing Step 2 outputs"
    )
    parser.add_argument(
        "--images-dir", type=str,
        default=str(PROJECT_ROOT / "datasets" / "RocasMaize" / "images" / "all"),
        help="Directory containing full original input images"
    )
    parser.add_argument(
        "--crop-images-dir", type=str,
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step3" / "cropped_images"),
        help="Directory where cropped plant images will be saved"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step3"),
        help="Directory where output dataset JSON files will be saved"
    )

    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    img_dir = Path(args.images_dir)
    crop_img_dir = Path(args.crop_images_dir)
    out_dir = Path(args.output_dir)

    results = main(
        input_source=in_dir,
        images_dir=img_dir,
        crop_images_output_dir=crop_img_dir
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for subset_name, subset_data in results.items():
        out_path = out_dir / f"{subset_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {subset_name}.json with {len(subset_data['items'])} cropped plant items to {out_path}")