"""Evaluate the two-stage pipeline: Mask2Former masks plus MTL BBCH prediction.
 
Stage 1: Mask2Former predicts the maize masks on the full images (maize_class, or
bbch_classes with --use_baseline_masks, whose BBCH class is then ignored).
Stage 2: Each predicted mask is cut out (see cut_maize_predictions.py) and the
trained MTL model predicts its BBCH stage with the regression and/or the
classification head.
 
The combined predictions are compared with the GT test set. For each BBCH head the
script prints the BBCH metrics per class (for all plants and separately for plants
with and without visible damage), saves the predictions for FiftyOne in
datasets/Predictions and saves a confusion matrix. Then the localization metrics
of stage 1 are printed. All metrics are logged to MLflow (experiment
"two_stage_evaluation").
 
The run config given with --config must contain "mode": "eval".
 
Usage (run_dir is the folder of a trained MTL run):
    python3 src/evaluation/two_stage_evaluation.py --config <run_dir>/config.json
    (add --use_baseline_masks to use the masks of the bbch_classes model)
"""


import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

for p in (SRC_DIR, SRC_DIR / "MTL", SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

GT_PATH = PROJECT_ROOT / "datasets/RocasMaize_BBCH_processed/mask2former/mask2former_maize_class/annotations/test.json"
PRED_OUT_ROOT = PROJECT_ROOT / "datasets/Predictions"
IOU_THRESHOLD = 0.5
BBCH_STAGES = list(range(10, 18))
BBCH_TASKS = ("bbch_regression", "bbch_classification")

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", type=str)
_pre.add_argument("--use_baseline_masks", action="store_true",
                  help="Use the masks predicted by the Mask2Former baseline (bbch_classes) instead of maize_class")
_pre_args, _rest = _pre.parse_known_args()
_config_path = Path(_pre_args.config)
if not _config_path.is_absolute():
    _config_path = PROJECT_ROOT / _config_path
with open(_config_path, "r") as _f:
    if json.load(_f).get("mode") != "eval":
        sys.exit(f"Error: {_config_path} requires \"mode\": \"eval\".")
sys.argv = [sys.argv[0], "--config", str(_config_path)] + _rest

if _pre_args.use_baseline_masks:
    STAGE1_DIR = PRED_OUT_ROOT / "mask2former_bbch_classes_pred"
    S1_SUFFIX = "_baseline_masks"
else:
    STAGE1_DIR = PRED_OUT_ROOT / "mask2former_maize_class_pred"
    S1_SUFFIX = ""
STAGE1_PRED = STAGE1_DIR / "annotations_fullres" / "test.json"
CUT_DIR = STAGE1_DIR / "cut_images"
CUT_ANN = CUT_DIR / "annotations.json"

from LibMTL.config import LibMTL_args  
from LibMTL.utils import set_random_seed  
from utils import parse_args, prepare_args, get_task_components  

PARAMS = parse_args(LibMTL_args)
set_random_seed(PARAMS.seed)

import mlflow 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from augmentations import get_transforms  
from bbch_dataset import BBCH_Dataset  
from bbch_trainer import BBCH_Trainer  
from decoders import get_decoder  
from encoders import get_encoder  
from metrics import InstanceDetectionMetrics, LocalizationMetrics  
from visualisations.confusion_matrix import plot_confusion_matrices_from_json  
import cut_maize_predictions as cut
from damage_subsets import evaluate_damage_subsets, _greedy_pairs, _is_damaged


def build_model(params):
    """Build the MTL model from the run config and load the trained checkpoint.
 
    One decoder is created per selected task. The checkpoint at params.load_path is
    loaded and the model is set to eval mode. Raises RuntimeError if the checkpoint
    is missing weights of the model.
 
    Returns the BBCH_Trainer with the loaded model.
    """
    kwargs, optim_param, scheduler_param = prepare_args(params)

    decoders = nn.ModuleDict()
    task_dict = {}
    for task in params.selected_tasks:
        decoder_type = params.task_definition.get(task, {}).get("decoder")
        num_classes = None
        if "classification" in task:
            num_classes = 8 if "bbch" in task else 2
        elif task == "semantic_segmentation":
            num_classes = 4
        decoders[task] = get_decoder(task_name=task, num_classes=num_classes, decoder_name=decoder_type)
        task_dict[task] = get_task_components(task, params)

    trainer = BBCH_Trainer(
        task_dict=task_dict, weighting=params.weighting, architecture=params.arch,
        encoder_class=lambda: get_encoder(params), decoders=decoders,
        rep_grad=params.rep_grad, multi_input=params.multi_input,
        optim_param=optim_param, scheduler_param=scheduler_param,
        save_path=params.save_path, load_path=params.load_path, params=params, **kwargs
    )

    checkpoint = torch.load(params.load_path, map_location=trainer.device)
    missing, _ = trainer.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint does not match the model, missing keys: {missing[:5]} ...")
    trainer.model.eval()
    return trainer


@torch.no_grad()
def predict_bbch(trainer, loader, params, bbch_tasks, bbch_map):
    """Predict the BBCH stage of all cut patches with each BBCH head.
 
    The regression output is un-normalized with bbch_min and bbch_max, rounded and
    clipped to the BBCH range. The classification output is mapped from the class
    index back to the BBCH stage with bbch_map.
 
    Returns a dict that maps each BBCH task to a list of BBCH stages, in the order
    of the loader.
    """
    idx_to_bbch = {v: int(k) for k, v in bbch_map.items()}
    preds = {t: [] for t in bbch_tasks}

    for images, _ in tqdm(loader, desc="Inference"):
        out = trainer.model(images.to(trainer.device))
        if "bbch_regression" in preds:
            raw = out["bbch_regression"].cpu().numpy() * (params.bbch_max - params.bbch_min) + params.bbch_min
            preds["bbch_regression"] += np.clip(np.rint(raw.flatten()).astype(int),
                                                int(params.bbch_min), int(params.bbch_max)).tolist()
        if "bbch_classification" in preds:
            idx = out["bbch_classification"].argmax(dim=-1).cpu().numpy()
            preds["bbch_classification"] += [idx_to_bbch[int(i)] for i in idx]
    return preds


def bbch_categories():
    """Build a category schema with one class per BBCH stage (10 to 17).
 
    Returns the categories dict.
    """
    return {"label": {"labels": [{"name": str(b), "attributes": []} for b in BBCH_STAGES], "attributes": []}}


def gt_as_bbch_classes(gt_data):
    """Convert the maize_class GT into BBCH classes.
 
    Each maize mask gets the label_id of its BBCH stage (from the bbch attribute).
 
    Returns a converted copy of the GT.
    """
    gt = copy.deepcopy(gt_data)
    gt["categories"] = bbch_categories()
    for item in gt["items"]:
        for ann in item.get("annotations", []):
            if ann.get("type") == "mask":
                ann["label_id"] = BBCH_STAGES.index(int(ann["attributes"]["bbch"]))
    return gt


def preds_as_bbch_classes(pred_data, patch_items, bbch_preds):
    """Give each stage 1 mask the BBCH class that was predicted for its patch.
 
    The patches are linked to the masks via image_id and pred_index. Masks without
    a patch (e.g. empty masks) are dropped.
 
    Returns the predictions with BBCH classes as a new dict.
    """
    label_of = {(it["image_id"], it["pred_index"]): BBCH_STAGES.index(b)
                for it, b in zip(patch_items, bbch_preds)}
    pred = {k: v for k, v in pred_data.items() if k != "items"}
    pred["categories"] = bbch_categories()
    pred["items"] = []
    for item in pred_data["items"]:
        masks = [a for a in item.get("annotations", []) if a.get("type") == "mask"]
        anns = []
        for p_idx, ann in enumerate(masks):
            if (item["id"], p_idx) in label_of:
                anns.append({**ann, "id": len(anns), "label_id": label_of[(item["id"], p_idx)]})
        pred["items"].append({**item, "annotations": anns})
    return pred


def with_gt_damage(pred_bbch, gt_data):
    """Add the damage_visible value of the matched GT plant to each predicted mask.
 
    Used for the FiftyOne export. The masks are matched like in the evaluation (see
    _greedy_pairs). Masks without a matched GT plant get None.
 
    Returns a copy of the predictions with the new attribute gt_damage_visible.
    """
    gt_by_id = {it["id"]: it for it in gt_data["items"]}
    out = copy.deepcopy(pred_bbch)
    out["categories"]["label"]["attributes"] = ["gt_damage_visible"]
    for item in out["items"]:
        P = [a for a in item["annotations"] if a.get("type") == "mask"]
        G = [a for a in gt_by_id.get(item["id"], {}).get("annotations", []) if a.get("type") == "mask"]
        for a in P:
            a["attributes"] = {**a.get("attributes", {}), "gt_damage_visible": None}
        for p, g in _greedy_pairs([a["rle"] for a in P], [a["rle"] for a in G], IOU_THRESHOLD):
            P[p]["attributes"]["gt_damage_visible"] = _is_damaged(G[g])
    return out


def main():
    """Run the two-stage evaluation and log the results to MLflow.
 
    Cuts the stage 1 masks if this was not done yet, predicts the BBCH stages and
    prints and saves the evaluation results. Exits if the run has no BBCH head.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
    warnings.filterwarnings("ignore", message=".*ShiftScaleRotate is a special case of Affine.*")
    params = PARAMS

    bbch_tasks = [t for t in BBCH_TASKS if t in params.selected_tasks]
    if not bbch_tasks:
        sys.exit("Error: The run has no BBCH head.")

    if not CUT_ANN.exists():
        print(f"{CUT_DIR} is missing or incomplete, cut images ...")
        cut.cut_predictions(STAGE1_PRED, CUT_DIR)

    dataset = BBCH_Dataset(ann_path=str(CUT_ANN), img_dir=str(CUT_DIR), params=params,
                           transform=get_transforms(params, is_training=False), is_training=False)

    batch_size = params.test_bs
    if params.img_resize == "gsd":
        largest = max(max(it["image"]["size"]) for it in dataset.items)
        if largest > params.global_scale:
            batch_size = 1

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=params.num_workers, pin_memory=True)

    trainer = build_model(params)
    preds = predict_bbch(trainer, loader, params, bbch_tasks, dataset.bbch_map)

    with open(STAGE1_PRED, "r") as f:
        pred_data = json.load(f)
    with open(GT_PATH, "r") as f:
        gt_data = json.load(f)
    gt_bbch = gt_as_bbch_classes(gt_data)

    run_dir = Path(params.save_path)
    ckpt_name = Path(params.load_path).stem
    vis_dir = run_dir.parent / "visualisations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}

    for task in bbch_tasks:
        pred_bbch = preds_as_bbch_classes(pred_data, dataset.items, preds[task])
        class_stats, macro_f1, weighted_f1, micro_f1, macro_mae, weighted_mae, micro_mae, macro_bias, weighted_bias, micro_bias = InstanceDetectionMetrics(iou_threshold=IOU_THRESHOLD).evaluate(pred_bbch, gt_bbch)

        prefix = f"{task}/" if len(bbch_tasks) > 1 else ""
        metrics[f"{prefix}macro_f1"] = float(macro_f1)
        for stats in class_stats.values():
            metrics[f"{prefix}f1_BBCH-{stats['name']}"] = float(stats["f1"])

        head = f" - {task}" if len(bbch_tasks) > 1 else ""
        print()
        print("=" * 105)
        print(f" BBCH EVALUATION (IoU-Threshold: {IOU_THRESHOLD}){head}")
        print("=" * 105)
        print(f"{'Class':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'MAE':<10} | {'BIAS':<10}")
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

        evaluate_damage_subsets(pred_bbch, gt_bbch, IOU_THRESHOLD)

        if len(bbch_tasks) > 1:
            print()

        suffix = (f"_{task}" if len(bbch_tasks) > 1 else "") + S1_SUFFIX

        pred_out = PRED_OUT_ROOT / f"two_stage_{run_dir.name}_{ckpt_name}{suffix}" / "annotations_fullres" / "test.json"
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_out, "w") as f:
            json.dump(with_gt_damage(pred_bbch, gt_data), f)
        print(f"\nStored Predictions: {pred_out}")

        plot_confusion_matrices_from_json(
            pred_data=pred_bbch,
            gt_data=gt_bbch,
            save_path=vis_dir / f"{run_dir.name}_{ckpt_name}{suffix}_two_stage_confusion_matrix.png",
            iou_threshold=IOU_THRESHOLD
        )

    loc = LocalizationMetrics(iou_threshold=IOU_THRESHOLD).evaluate(pred_data, gt_data)
    for k in ("tp", "fp", "fn", "precision", "recall", "f1", "mean_iou"):
        metrics[f"localisation_{k}"] = float(loc[k])

    print("\n" + "=" * 105)
    print(" LOCALIZATION EVALUATION")
    print("=" * 105)
    print(f"True Positives (correctly detected maize plants): {loc['tp']}")
    print(f"False Positives (hallucinated maize plants):  {loc['fp']}")
    print(f"False Negatives (missed maize plants):   {loc['fn']}")
    print("-" * 105)
    print(f"Localization Precision: {loc['precision']:.4f}")
    print(f"Localization Recall:    {loc['recall']:.4f}")
    print(f"Localization F1-Score:  {loc['f1']:.4f}")
    print(f"Mean Mask-IoU (only TPs): {loc['mean_iou']:.4f}\n")

    mlflow.set_tracking_uri(f"sqlite:///{(run_dir.parent / 'mlflow.db').resolve()}")
    mlflow.set_experiment("two_stage_evaluation")
    with mlflow.start_run(run_name=f"{run_dir.name}__{ckpt_name}{S1_SUFFIX}"):
        mlflow.log_params({"run": run_dir.name, "checkpoint": ckpt_name, "stage1_pred": str(STAGE1_PRED)})
        mlflow.log_metrics(metrics)



if __name__ == "__main__":
    main()