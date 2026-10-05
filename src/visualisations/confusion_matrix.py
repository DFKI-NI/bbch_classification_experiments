"""Plot the confusion matrix of predicted instance classes (e.g. BBCH classes).
 
The predicted and GT masks are matched by mask IoU (see
InstanceDetectionMetrics.get_matched_pairs). Only matched masks are used, so missed
plants and false detections are not part of the matrix. The matrix is normalized
per row, so each row shows how the plants of one GT class were predicted.
 
This module is used by the evaluation scripts and has no command line entry point.
"""
 

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from metrics import InstanceDetectionMetrics


def _draw_single_cm(gts, preds, labels, save_path, title, cmap="Blues"):
    """Draw one row-normalized confusion matrix and save it as an image.
 
    Each cell shows the share of the row and the absolute count. The diagonal cells
    also show the number of samples of the class. The y-axis labels show the number
    of GT samples per class. Nothing is drawn if gts is empty.
    """
    if len(gts) == 0:
        print("[WARNING] No data for drawing confusion matrix found.")
        return

    cm_abs = confusion_matrix(gts, preds, labels=labels)
    
    row_sums = cm_abs.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(
        cm_abs.astype('float'), 
        row_sums, 
        out=np.zeros_like(cm_abs, dtype=float), 
        where=row_sums != 0
    )

    y_axis_labels = [f"{label} (n={int(np.sum(cm_abs[i, :]))})" for i, label in enumerate(labels)]

    cell_annots = []
    for i in range(len(labels)):
        row = []
        n_total_class = np.sum(cm_abs[i, :])
        for j in range(len(labels)):
            perc = cm_norm[i, j]
            abs_val = int(cm_abs[i, j])
            if i == j:
                row.append(f"{perc:.3f}\n({abs_val}/{int(n_total_class)})")
            else:
                row.append(f"{perc:.3f}\n({abs_val})")
        cell_annots.append(row)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, 
        annot=np.array(cell_annots), 
        fmt="",                      
        cmap=cmap,
        xticklabels=labels, 
        yticklabels=y_axis_labels,   
        annot_kws={"size": 10}       
    )
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel("Ground Truth BBCH (n = Number of Samples)", fontsize=12)
    plt.xlabel("Predicted BBCH", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrices_from_json(pred_data, gt_data, save_path, iou_threshold=0.5):
    """Match the predicted and GT masks by IoU and save their confusion matrix.
 
    The classes on the axes are all classes that appear in the matched pairs. The
    folder of save_path is created if needed. Nothing is saved if no masks match.
    """
    metric = InstanceDetectionMetrics(iou_threshold=iou_threshold)
    matched_pairs = metric.get_matched_pairs(pred_data, gt_data)

    if not matched_pairs:
        print("[WARNING] No overlaps for confusion matrix found.")
        return

    gts = [pair[0] for pair in matched_pairs]
    preds = [pair[1] for pair in matched_pairs]

    labels = sorted(list(set(gts) | set(preds)))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    _draw_single_cm(
        gts=gts,
        preds=preds,
        labels=labels,
        save_path=save_path,
        title="Normalised Confusion Matrix (BBCH Classes)",
        cmap="Blues"
    )
    print(f"Stored Confusion Matrix at: {save_path}")