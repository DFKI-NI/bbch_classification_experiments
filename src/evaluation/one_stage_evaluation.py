"""Evaluate the one-stage Mask2Former BBCH predictions against the ground truth.
 
The one-stage model (mask2former_bbch_classes) predicts the maize masks and the
BBCH class in one step. This script compares its predictions with the GT test set
and prints:
 
1. The BBCH metrics per class (TP, FP, FN, precision, recall, F1, MAE and BIAS)
   and the macro and micro scores, for all plants and separately for plants with
   and without visible damage.
2. The localization metrics (detection of the maize plants and mean mask IoU).
 
The results are logged to the MLflow database (experiment "one_stage_evaluation").
Unless --no-plot is set, the confusion matrix is saved in
training_runs/visualisations.
 
Usage:
    python3 src/evaluation/one_stage_evaluation.py
"""


import argparse
from datetime import datetime
import json
import sys
from pathlib import Path
import mlflow


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent 
DEFAULT_DB_PATH = PROJECT_ROOT / "training_runs" / "mlflow.db"
VIS_DIR_PATH = PROJECT_ROOT / "training_runs" / "visualisations"
METRICS_DIR = SCRIPT_DIR.parent
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))

from metrics import InstanceDetectionMetrics, LocalizationMetrics
from damage_subsets import evaluate_damage_subsets
from visualisations.confusion_matrix import plot_confusion_matrices_from_json


def log_results_to_mlflow(run_timestamp, macro_f1, class_stats, loc_results, pred_data, gt_data, iou_threshold, should_plot):
    """Log the evaluation results to the central MLflow database.
 
    Logs the localization metrics, the macro F1 and the F1 of each BBCH class in a
    new run of the experiment "one_stage_evaluation". If should_plot is True, the
    confusion matrix is also saved as {run_timestamp}_confusion_matrix.png in
    VIS_DIR_PATH.
    """
    mlflow.set_tracking_uri(f"sqlite:///{DEFAULT_DB_PATH.resolve()}")
    mlflow.set_experiment("one_stage_evaluation")

    with mlflow.start_run(run_name=run_timestamp):
        metrics_to_log = {
            "localisation_tp": int(loc_results["tp"]),
            "localisation_fp": int(loc_results["fp"]),
            "localisation_fn": int(loc_results["fn"]),
            "localisation_precision": float(loc_results["precision"]),
            "localisation_recall": float(loc_results["recall"]),
            "localisation_f1": float(loc_results["f1"]),
            "localisation_mean_iou": float(loc_results["mean_iou"]),
            "macro_f1": float(macro_f1),
        }

        for c_id, stats in class_stats.items():
            class_name = stats.get("name", f"class_{c_id}")
            metrics_to_log[f"f1_BBCH-{class_name}"] = float(stats["f1"])

        mlflow.log_metrics(metrics_to_log)

        if should_plot:
            plot_file_name = f"{run_timestamp}_confusion_matrix.png"
            plot_save_path = VIS_DIR_PATH / plot_file_name
            plot_confusion_matrices_from_json(
                pred_data=pred_data, 
                gt_data=gt_data, 
                save_path=plot_save_path,
                iou_threshold=iou_threshold
            )


def resolve_path(path_str: str) -> Path:
    """Convert a path string to an absolute Path.
 
    Relative paths are resolved from the project root.
 
    Returns the absolute path.
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_args():
    """Parse the command line arguments.
 
    Returns the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluation of the mask2former with 8 BBCH classes")
    parser.add_argument(
        "--pred",
        type=str,
        default="datasets/Predictions/mask2former_bbch_classes_pred/annotations_fullres/test.json",
        help="Path to PRED JSON"
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="datasets/RocasMaize_BBCH_processed/mask2former/mask2former_bbch_classes/annotations/test.json",
        help="Path to GT JSON"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for mask matching (default: 0.5)"
    )
    
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot confusion matrix (default: --plot, no plot via --no-plot)"
    )
    return parser.parse_args()


def main():
    """Load the predictions and the GT, run all evaluations and log the results.
 
    Raises FileNotFoundError if the prediction or GT file does not exist.
    """
    args = parse_args()

    pred_path = resolve_path(args.pred)
    gt_path = resolve_path(args.gt)

    if not pred_path.exists():
        raise FileNotFoundError(f"Pred file not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    with open(pred_path, "r") as f:
        pred_data = json.load(f)
    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    run_timestamp = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    bbch_metric = InstanceDetectionMetrics(iou_threshold=args.iou_threshold)
    class_stats, macro_f1, weighted_f1, micro_f1, macro_mae, weighted_mae, micro_mae, macro_bias, weighted_bias, micro_bias = bbch_metric.evaluate(pred_data, gt_data)

    print("=" * 105)
    print(f" BBCH EVALUATION (IoU-Threshold: {args.iou_threshold})")
    print("=" * 105)
    print(f"{'Class':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'MAE':<10} | {'BIAS':<10}")
    print("-" * 105)

    for c_id, stats in class_stats.items():
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
    #print(f"WEIGHTED F1-SCORE: {weighted_f1:.4f}")
    print(f"MICRO F1-SCORE: {micro_f1:.4f}")
    print(f"MACRO MAE: {macro_mae:.4f}")
    #print(f"WEIGHTED MAE-SCORE: {weighted_mae:.4f}")
    print(f"MICRO MAE: {micro_mae:.4f}")
    print(f"MACRO BIAS: {macro_bias:.4f}")
    #print(f"WEIGHTED BIAS-SCORE: {weighted_bias:.4f}")
    print(f"MICRO BIAS: {micro_bias:.4f}")

    evaluate_damage_subsets(pred_data, gt_data, args.iou_threshold)

    loc_metric = LocalizationMetrics(iou_threshold=args.iou_threshold)
    loc_results = loc_metric.evaluate(pred_data, gt_data)

    print("\n" + "=" * 105)
    print(" LOCALIZATION EVALUATION")
    print("=" * 105)
    print(f"True Positives (correctly detected maize plants): {loc_results['tp']}")
    print(f"False Positives (hallucinated maize plants):  {loc_results['fp']}")
    print(f"False Negatives (missed maize plants):   {loc_results['fn']}")
    print("-" * 105)
    print(f"Localization Precision: {loc_results['precision']:.4f}")
    print(f"Localization Recall:    {loc_results['recall']:.4f}")
    print(f"Localization F1-Score:  {loc_results['f1']:.4f}")
    print(f"Mean Mask-IoU (only TPs): {loc_results['mean_iou']:.4f}\n")
    
    log_results_to_mlflow(
        run_timestamp=run_timestamp,
        macro_f1=macro_f1,
        class_stats=class_stats,
        loc_results=loc_results,
        pred_data=pred_data,
        gt_data=gt_data,
        iou_threshold=args.iou_threshold,
        should_plot=args.plot
    )


if __name__ == "__main__":
    main()