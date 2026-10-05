"""Cut the predicted maize masks of Mask2Former out of the full images.
 
This script reads the full-resolution Mask2Former predictions (maize_class) and
crops one image patch per predicted maize mask, using the bounding box of the
mask. The patches are saved as PNG files in CUT_DIR, together with an
annotations.json file that has one item per patch.
 
Each item gets placeholder attributes (bbch = 10.0, ppr_cm = 0.0,
damage_visible = False), because the true values of a prediction are not known.
The fields image_id and pred_index link each patch back to its full image and to
the prediction. CUT_DIR is deleted and created again on each run.
 
Usage:
    python3 src/evaluation/cut_maize_predictions.py
"""


import json
import shutil
from pathlib import Path
import cv2
from pycocotools import mask as mask_utils
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PRED_PATH = PROJECT_ROOT / "datasets/Predictions/mask2former_maize_class_pred/annotations_fullres/test.json"
IMAGES_DIR = PROJECT_ROOT / "datasets/RocasMaize/images/all"
CUT_DIR = PROJECT_ROOT / "datasets/Predictions/mask2former_maize_class_pred/cut_images"


def get_bbox_from_rle(rle):
    """Compute the bounding box of an RLE mask.
 
    Returns (x1, y1, x2, y2) in pixels. x2 and y2 are exclusive, so they can be
    used directly for slicing.
    """
    x, y, w, h = [int(round(v)) for v in mask_utils.toBbox(rle)]
    return x, y, x + w, y + h


def make_item(patch_id, crop_h, crop_w, image_id, pred_index):
    """Create a dataset item for one cut prediction patch.
 
    The annotation only has placeholder attributes, because the true values of a
    prediction are not known.
 
    Returns the item as a dict.
    """
    filename = f"{patch_id}.png"
    return {
        "id": patch_id,
        "image": {"file_name": filename, "path": filename, "size": [crop_h, crop_w]},
        "annotations": [{
            "id": 0, "type": "mask", "label_id": 0,
            "attributes": {"is_main_plant": True, "bbch": 10.0, "ppr_cm": 0.0, "damage_visible": False},
        }],
        "image_id": image_id,
        "pred_index": pred_index,
    }


def cut_predictions(pred_path=PRED_PATH, cut_dir=CUT_DIR):
    """Cut all predicted maize masks out of the full images and save the patches.
 
    cut_dir is deleted first, so old patches are removed. Predictions with an
    empty mask are skipped. Raises FileNotFoundError if a full image is missing.
    """
    with open(pred_path, "r") as f:
        pred_data = json.load(f)

    if cut_dir.exists():
        shutil.rmtree(cut_dir)
    cut_dir.mkdir(parents=True)

    items = []
    for pred_item in tqdm(pred_data["items"], desc="Cutting images"):
        img_id = pred_item["id"]
        pred_anns = [a for a in pred_item.get("annotations", []) if a.get("type") == "mask"]
        if not pred_anns:
            continue

        image = cv2.imread(str(IMAGES_DIR / f"{img_id}.png"))
        if image is None:
            raise FileNotFoundError(f"Image not found: {IMAGES_DIR / f'{img_id}.png'}")

        for p_idx, p_ann in enumerate(pred_anns):
            if mask_utils.area(p_ann["rle"]) == 0:
                continue
            x1, y1, x2, y2 = get_bbox_from_rle(p_ann["rle"])
            patch_id = f"{img_id}_pred_{p_idx}"
            cv2.imwrite(str(cut_dir / f"{patch_id}.png"), image[y1:y2, x1:x2])
            items.append(make_item(patch_id, y2 - y1, x2 - x1, img_id, p_idx))

    with open(cut_dir / "annotations.json", "w") as f:
        json.dump({"items": items}, f, indent=2)

    print(f"{len(items)} predicted masks were cut and stored in -> {cut_dir}\n")



if __name__ == "__main__":
    cut_predictions()