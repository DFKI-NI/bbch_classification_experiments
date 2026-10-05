"""Metrics for the MTL training and for the instance evaluation.
 
The first classes are LibMTL metrics (AbsMetric) that are updated per batch during
training: update_fun collects the predictions of one batch, score_fun computes the
final scores at the end of an epoch and reinit resets the metric for the next
epoch or split.
 
InstanceDetectionMetrics and LocalizationMetrics evaluate predicted instance masks
(RLE) against the GT. The masks are matched one-to-one by mask IoU, highest IoU
first, and a match needs at least iou_threshold.
"""


from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
import numpy as np
import torch
import torch.nn.functional as F
from LibMTL.metrics import AbsMetric
from pycocotools import mask as mask_utils


class F1Metrics(AbsMetric):
    """Macro F1 score for a multi-class classification (e.g. bbch_classification)."""
    def __init__(self, num_classes=8):
        super(F1Metrics, self).__init__()
        self.num_classes = num_classes
        self.all_preds = []
        self.all_gts = []
        
    def update_fun(self, pred, gt):
        """Collect the predicted and GT class indices of one batch."""
        pred_idx = F.softmax(pred, dim=-1).max(-1)[1].cpu().numpy()
        gt_idx = gt.cpu().numpy()

        self.all_preds.extend(pred_idx)
        self.all_gts.extend(gt_idx)

        self.record.append(np.sum(pred_idx == gt_idx))
        self.bs.append(len(gt_idx))
        
    def score_fun(self):
        """Compute the macro F1 over all collected samples.
 
        Returns [macro_f1].
        """
        macro_f1 = f1_score(self.all_gts, self.all_preds, average='macro', zero_division=0)
        return [macro_f1]

    def reinit(self):
        """Reset the collected predictions and GT values."""
        super(F1Metrics, self).reinit()
        self.all_preds = []
        self.all_gts = []


class F1RegressionMetrics(AbsMetric):
    """Macro F1 score for the BBCH regression.
 
    The normalized predictions and GT values are scaled back to the BBCH range
    (bbch_min to bbch_max) and rounded to the nearest stage, so the regression can
    be compared with the classification.
    """

    def __init__(self, bbch_min=10.0, bbch_max=17.0):
        super(F1RegressionMetrics, self).__init__()
        self.bbch_min = bbch_min
        self.bbch_max = bbch_max
        self.all_preds_raw = []
        self.all_gts_raw = []
        
    def update_fun(self, pred, gt):
        """Collect the predictions and GT values of one batch in the BBCH range."""
        # Scale back to the original range [10, 17]
        pred_raw = pred.detach().cpu().numpy() * (self.bbch_max - self.bbch_min) + self.bbch_min
        gt_raw = gt.detach().cpu().numpy() * (self.bbch_max - self.bbch_min) + self.bbch_min
        
        self.all_preds_raw.extend(pred_raw.flatten())
        self.all_gts_raw.extend(gt_raw.flatten())
        
        self.record.append(np.sum(np.abs(pred_raw - gt_raw)))
        self.bs.append(len(gt_raw))
        
    def score_fun(self):
        """Round the predictions to BBCH stages and compute the macro F1.
 
        The rounded predictions are clipped to bbch_min and bbch_max.
 
        Returns [macro_f1].
        """
        preds = np.array(self.all_preds_raw)
        gts = np.array(self.all_gts_raw)
        
        preds_rounded = np.rint(preds).astype(int)
        preds_rounded = np.clip(preds_rounded, int(self.bbch_min), int(self.bbch_max))
        
        gts_rounded = np.rint(gts).astype(int)
        f1 = f1_score(gts_rounded, preds_rounded, average='macro', zero_division=0)
  
        return [f1]

    def reinit(self):
        """Reset the collected predictions and GT values."""
        super(F1RegressionMetrics, self).reinit()
        self.all_preds_raw = []
        self.all_gts_raw = []


class PPRCMMetrics(AbsMetric):
    """RMSE and MAE for the ppr_cm regression, in cm.
 
    The normalized values are scaled back to the range ppr_min to ppr_max. The GT
    can also be a [ppr_cm, weight] tensor, then only the first column is used.
    """

    def __init__(self, ppr_min=0.0, ppr_max=3.0):
        super(PPRCMMetrics, self).__init__()
        self.ppr_min = ppr_min
        self.ppr_max = ppr_max
        self.all_preds_raw = []
        self.all_gts_raw = []
        
    def update_fun(self, pred, gt):
        """Collect the predictions and GT values of one batch in cm."""
        if gt.dim() > 1 and gt.size(1) == 2:
            gt_val = gt[:, 0:1]
        else:
            gt_val = gt

        # Denormalize from [0, 1] back to [0, 3] cm
        pred_raw = pred.detach().cpu().numpy() * (self.ppr_max - self.ppr_min) + self.ppr_min
        gt_raw = gt_val.detach().cpu().numpy() * (self.ppr_max - self.ppr_min) + self.ppr_min
        
        self.all_preds_raw.extend(pred_raw.flatten())
        self.all_gts_raw.extend(gt_raw.flatten())
        
        self.record.append(np.sum(np.abs(pred_raw - gt_raw)))
        self.bs.append(len(gt_raw))
        
    def score_fun(self):
        """Compute RMSE and MAE over all collected samples.
 
        Returns [rmse, mae].
        """
        preds = np.array(self.all_preds_raw)
        gts = np.array(self.all_gts_raw)
        mae = mean_absolute_error(gts, preds)
        rmse = np.sqrt(mean_squared_error(gts, preds))
        return [rmse, mae]

    def reinit(self):
        """Reset the collected predictions and GT values."""
        super(PPRCMMetrics, self).reinit()
        self.all_preds_raw = []
        self.all_gts_raw = []


class F1BinaryMetrics(AbsMetric):
    """Macro F1 score for a binary classification (e.g. damaged_classification)."""
    def __init__(self):
        super(F1BinaryMetrics, self).__init__()
        self.all_preds = []
        self.all_gts = []
        
    def update_fun(self, pred, gt):
        """Collect the predicted and GT class indices of one batch."""
        pred_idx = F.softmax(pred, dim=-1).argmax(-1).cpu().numpy()
        gt_idx = gt.cpu().numpy()

        self.all_preds.extend(pred_idx)
        self.all_gts.extend(gt_idx)

        self.record.append(np.sum(pred_idx == gt_idx))
        self.bs.append(len(gt_idx))
        
    def score_fun(self):
        """Compute the macro F1, the mean of the F1 of both classes.
 
        Returns [macro_f1].
        """
        macro_f1 = f1_score(self.all_gts, self.all_preds, average='macro', zero_division=0)
        f1_per_class = f1_score(self.all_gts, self.all_preds, average=None, zero_division=0)
        return [macro_f1]

    def reinit(self):
        """Reset the collected predictions and GT values."""
        super(F1BinaryMetrics, self).reinit()
        self.all_preds = []
        self.all_gts = []



class SegmentationMetrics(AbsMetric):
    """mIoU, macro F1 (Dice) and main plant IoU for the semantic segmentation.
 
    The scores are computed from a confusion matrix that is collected over all
    batches. Pixels with ignore_index are skipped. Class 3 is the main plant (see
    step4_create_semantic_masks.py).
    """

    def __init__(self, num_classes=4, ignore_index=255):
        super(SegmentationMetrics, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        self._printed_debug = False
        
    def update_fun(self, pred, gt):
        """Add the pixels of one batch to the confusion matrix."""
        pred_idx = torch.argmax(pred, dim=1).detach().cpu().numpy()
        gt_idx = gt.detach().cpu().numpy()

        mask = gt_idx != self.ignore_index
        valid_preds = pred_idx[mask]
        valid_gts = gt_idx[mask]

        if len(valid_gts) > 0:
            self.confusion_matrix += np.bincount(
                self.num_classes * valid_gts.astype(int) + valid_preds.astype(int),
                minlength=self.num_classes**2
            ).reshape(self.num_classes, self.num_classes)

            correct = np.sum(valid_preds == valid_gts)
            self.record.append(correct)
            self.bs.append(len(valid_gts))
        
    def score_fun(self):
        """Compute the scores from the confusion matrix.
 
        Returns [miou, macro_f1, main_plant_iou].
        """
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        
        iou_denom = tp + fp + fn
        iou_per_class = np.divide(tp, iou_denom, out=np.zeros_like(tp, dtype=float), where=iou_denom != 0)
        miou = np.mean(iou_per_class)

        f1_denom = 2 * tp + fp + fn
        f1_per_class = np.divide(2 * tp, f1_denom, out=np.zeros_like(tp, dtype=float), where=f1_denom != 0)
        macro_f1 = np.mean(f1_per_class)
        
        if self.num_classes >= 4:
            plant_iou = iou_per_class[3]
        else:
            plant_iou = 0.0
            print(f"[WARN] plant_iou is 0.0 because num_classes ({self.num_classes}) < 4!")

        return [float(miou), float(macro_f1), float(plant_iou)]

    def reinit(self):
        """Reset the confusion matrix."""
        super(SegmentationMetrics, self).reinit()
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self._printed_debug = False

    

class DistanceRegressionMetrics(AbsMetric):
    """MAE and RMSE for the distance regression.
 
    The errors are computed for two sets of pixels: all valid pixels (GT above
    min_valid) and only the plant pixels (GT >= 0, the real distances of the
    main plant).
    """

    def __init__(self, min_valid=-1.5):
        super(DistanceRegressionMetrics, self).__init__()
        self.min_valid = min_valid
        self.all_errors = []
        self.plant_errors = []

    def update_fun(self, pred, gt):
        """Collect the errors of one batch for all valid pixels and plant pixels."""
        pred_flat = pred.detach().cpu().numpy().flatten()
        gt_flat = gt.detach().cpu().numpy().flatten()

        mask = gt_flat > self.min_valid
        if np.any(mask):
            diff = pred_flat[mask] - gt_flat[mask]
            self.all_errors.extend(diff.tolist())
            self.record.append(np.sum(np.abs(diff)))
            self.bs.append(len(diff))

        plant_mask = gt_flat >= 0
        if np.any(plant_mask):
            plant_diff = pred_flat[plant_mask] - gt_flat[plant_mask]
            self.plant_errors.extend(plant_diff.tolist())
        
    def score_fun(self):
        """Compute MAE and RMSE for all valid pixels and for the plant pixels.
 
        Returns [rmse, mae, plant_mae, plant_rmse]. Values that are 0 are returned
        as 1e-8.
        """
        def calc_metrics(err_list):
            """Compute MAE and RMSE of a list of errors.
 
            Returns (mae, rmse), or (1e-8, 1e-8) if the list is empty.
            """
            if not err_list:
                return 1e-8, 1e-8
            err_array = np.array(err_list)
            mae = np.mean(np.abs(err_array))
            rmse = np.sqrt(np.mean(err_array**2))
            return mae, rmse

        mae, rmse = calc_metrics(self.all_errors)
        p_mae, p_rmse = calc_metrics(self.plant_errors)
        
        results = [rmse, mae, p_mae, p_rmse]
        return [float(v) if v != 0.0 else 1e-8 for v in results]

    def reinit(self):
        """Reset the collected errors."""
        super(DistanceRegressionMetrics, self).reinit()
        self.all_errors = []
        self.plant_errors = []



class DirectionRegressionMetrics(AbsMetric):
    """Errors of the predicted direction vectors.
 
    Only pixels with a valid GT vector are used: the first value must be above
    min_valid and the vector length above 0, so background pixels ([0, 0]) are
    skipped.
    """

    def __init__(self, min_valid=-1.5):
        super(DirectionRegressionMetrics, self).__init__()
        self.min_valid = min_valid
        self.all_angular_errors = []
        self.all_magnitude_errors = []
        self.all_vector_mae = []

    def update_fun(self, pred, gt):
        """Collect the vector, angle and length errors of one batch.
 
        - vector error: Euclidean distance between the predicted and the GT vector
        - angle error: angle between both vectors in degrees
        - length error: difference of the vector lengths (the GT length is 1)
        """
        pred_flat = pred.detach().permute(0, 2, 3, 1).reshape(-1, 2)
        gt_flat = gt.detach().permute(0, 2, 3, 1).reshape(-1, 2)
        
        mask = (gt_flat[:, 0] > self.min_valid) & (torch.norm(gt_flat, dim=1) > 1e-4)
        
        if not mask.any():
            return

        p = pred_flat[mask]
        g = gt_flat[mask]

        p_norm = F.normalize(p, p=2, dim=1)
        g_norm = F.normalize(g, p=2, dim=1)
        dot_product = torch.sum(p_norm * g_norm, dim=1).clamp(-1.0, 1.0)
        angular_errors = torch.acos(dot_product) * (180.0 / np.pi)

        p_mag = torch.norm(p, p=2, dim=1)
        g_mag = torch.norm(g, p=2, dim=1)
        mag_errors = torch.abs(p_mag - g_mag)

        v_mae = torch.norm(p - g, p=2, dim=1)

        self.all_angular_errors.extend(angular_errors.cpu().numpy().tolist())
        self.all_magnitude_errors.extend(mag_errors.cpu().numpy().tolist())
        self.all_vector_mae.extend(v_mae.cpu().numpy().tolist())
        
        self.record.append(torch.sum(v_mae).cpu().item())
        self.bs.append(len(v_mae))

    def score_fun(self):
        """Compute the mean errors over all collected pixels.
 
        Returns [vec_error, ang_error, magni_error], or zeros if no pixel was valid.
        """
        if not self.all_vector_mae:
            return [0.0, 0.0, 0.0]
    
        return [
            np.mean(self.all_vector_mae),
            np.mean(self.all_angular_errors),
            np.mean(self.all_magnitude_errors)
        ]

    def reinit(self):
        """Reset the collected errors."""
        super(DirectionRegressionMetrics, self).reinit()
        self.all_angular_errors = []
        self.all_magnitude_errors = []
        self.all_vector_mae = []




class InstanceDetectionMetrics:
    """Evaluate predicted instance masks with classes (e.g. BBCH classes).
 
    A match with the same class is a TP. A match with a different class is an FP
    for the predicted class and an FN for the GT class. Unmatched predictions are
    FPs and unmatched GT masks are FNs.
    """

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def evaluate(self, pred_data, gt_data):
        """Compute the detection metrics per class and over all classes.
 
        MAE and BIAS are computed from the matched masks of each GT class as the
        (absolute) difference of the class indices, so in BBCH stages for BBCH
        classes. A BIAS above 0 means the stage is predicted too high. Class index 7
        (BBCH 17) is left out of all macro, weighted and micro scores. The weighted
        scores use the number of matched masks per class as weights.
 
        Returns class_stats (TP, FP, FN, precision, recall, F1, MAE and BIAS per
        class), macro_f1, weighted_f1, micro_f1, macro_mae, weighted_mae,
        micro_mae, macro_bias, weighted_bias and micro_bias.
        """
        gt_labels_list = gt_data["categories"]["label"]["labels"]
        class_id_to_name = {idx: label["name"] for idx, label in enumerate(gt_labels_list)}

        class_stats = {
            idx: {"tp": 0, "fp": 0, "fn": 0, "name": name,
                  "abs_error": [], "error": [], "count": 0}
            for idx, name in class_id_to_name.items()
        }

        gt_dict = {item["id"]: item["annotations"] for item in gt_data.get("items", [])}

        for pred_item in pred_data.get("items", []):
            image_id = pred_item.get("id")
            if image_id not in gt_dict:
                continue

            pred_anns = [a for a in pred_item.get("annotations", []) if a.get("type") == "mask"]
            gt_anns = [a for a in gt_dict[image_id] if a.get("type") == "mask"]

            if not pred_anns and gt_anns:
                for gt in gt_anns:
                    gt_label = gt["label_id"]
                    if gt_label in class_stats:
                        class_stats[gt_label]["fn"] += 1
                continue

            if pred_anns and not gt_anns:
                for pred in pred_anns:
                    pred_label = pred["label_id"]
                    if pred_label in class_stats:
                        class_stats[pred_label]["fp"] += 1
                continue

            if not pred_anns and not gt_anns:
                continue

            pred_rles = [p["rle"] for p in pred_anns]
            gt_rles = [g["rle"] for g in gt_anns]
            iou_matrix = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles))

            matched_gts = set()
            matched_preds = set()

            flat_indices = np.argsort(iou_matrix, axis=None)[::-1]

            for idx in flat_indices:
                p_idx, g_idx = np.unravel_index(idx, iou_matrix.shape)

                if p_idx in matched_preds or g_idx in matched_gts:
                    continue

                iou_val = iou_matrix[p_idx, g_idx]
                if iou_val < self.iou_threshold:
                    break

                matched_preds.add(p_idx)
                matched_gts.add(g_idx)

                pred_label = pred_anns[p_idx]["label_id"]
                gt_label = gt_anns[g_idx]["label_id"]

                class_stats[gt_label]["count"] += 1
                class_stats[gt_label]["abs_error"].append(abs(pred_label-gt_label))
                class_stats[gt_label]["error"].append(pred_label-gt_label)

                if pred_label == gt_label:
                    class_stats[gt_label]["tp"] += 1
                else:
                    class_stats[pred_label]["fp"] += 1
                    class_stats[gt_label]["fn"] += 1

            for p_idx in range(len(pred_anns)):
                if p_idx not in matched_preds:
                    pred_label = pred_anns[p_idx]["label_id"]
                    if pred_label in class_stats:
                        class_stats[pred_label]["fp"] += 1

            for g_idx in range(len(gt_anns)):
                if g_idx not in matched_gts:
                    gt_label = gt_anns[g_idx]["label_id"]
                    if gt_label in class_stats:
                        class_stats[gt_label]["fn"] += 1

        f1_scores_for_macro = []
        tp_global = 0
        fp_global = 0
        fn_global = 0
        mae_scores_for_macro = []
        mae_scores_for_micro = []
        ae_scores_for_macro = []
        ae_scores_for_micro = []
        class_weights = []

        for c_id, stats in class_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            if not c_id == 7:
                tp_global += tp
                fp_global += fp
                fn_global += fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            if not c_id == 7:
                mae_scores_for_micro.extend(stats["abs_error"])
                ae_scores_for_micro.extend(stats["error"])
                class_weights.append(len(stats["abs_error"]))
            mae = np.mean(stats["abs_error"])
            ae = np.mean(stats["error"])

            stats["precision"] = precision
            stats["recall"] = recall
            stats["f1"] = f1
            stats["MAE"] = mae
            stats["BIAS"] = ae
            if not c_id == 7:
                f1_scores_for_macro.append(f1)
                mae_scores_for_macro.append(mae)
                ae_scores_for_macro.append(ae)

        class_weight_sum = np.sum(class_weights)
        class_weights_norm = [a/class_weight_sum for a in class_weights]

        macro_f1 = float(np.mean(f1_scores_for_macro)) if f1_scores_for_macro else 0.0
        weighted_f1 = 0
        for class_f1, norm_class_weight in zip(f1_scores_for_macro, class_weights_norm):
            weighted_f1 += class_f1*norm_class_weight

        weighted_mae = 0
        for class_mae, norm_class_weight in zip(mae_scores_for_macro, class_weights_norm):
            weighted_mae += class_mae*norm_class_weight

        weighted_ae = 0
        for class_ae, norm_class_weight in zip(ae_scores_for_macro, class_weights_norm):
            weighted_ae += class_ae*norm_class_weight

        global_precision = tp_global / (tp_global + fp_global) if (tp_global + fp_global) > 0 else 0.0
        global_recall = tp_global / (tp_global + fn_global) if (tp_global + fn_global) > 0 else 0.0
        micro_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
        
        macro_mae = float(np.mean(mae_scores_for_macro)) if mae_scores_for_macro else 0.0
        micro_mae = float(np.mean(mae_scores_for_micro)) if mae_scores_for_micro else 0.0
        macro_ae = float(np.mean(ae_scores_for_macro)) if ae_scores_for_macro else 0.0
        micro_ae = float(np.mean(ae_scores_for_micro)) if ae_scores_for_micro else 0.0

        return class_stats, macro_f1, weighted_f1, micro_f1, macro_mae, weighted_mae, micro_mae, macro_ae, weighted_ae, micro_ae

    def get_matched_pairs(self, pred_data, gt_data):
        """Collect the class pairs of all matched masks for the confusion matrix.
 
        Uses the same IoU matching as evaluate.
 
        Returns a list of (gt_label, pred_label) tuples.
        """
        gt_dict = {item["id"]: item["annotations"] for item in gt_data.get("items", [])}
        
        matched_pairs = []

        for pred_item in pred_data.get("items", []):
            image_id = pred_item.get("id")
            if image_id not in gt_dict:
                continue

            pred_anns = [a for a in pred_item.get("annotations", []) if a.get("type") == "mask"]
            gt_anns = [a for a in gt_dict[image_id] if a.get("type") == "mask"]

            if not pred_anns or not gt_anns:
                continue

            pred_rles = [p["rle"] for p in pred_anns]
            gt_rles = [g["rle"] for g in gt_anns]
            iou_matrix = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles))

            matched_gts = set()
            matched_preds = set()

            flat_indices = np.argsort(iou_matrix, axis=None)[::-1]

            for idx in flat_indices:
                p_idx, g_idx = np.unravel_index(idx, iou_matrix.shape)

                if p_idx in matched_preds or g_idx in matched_gts:
                    continue

                iou_val = iou_matrix[p_idx, g_idx]
                if iou_val < self.iou_threshold:
                    break

                matched_preds.add(p_idx)
                matched_gts.add(g_idx)

                gt_label = gt_anns[g_idx]["label_id"]
                pred_label = pred_anns[p_idx]["label_id"]
                matched_pairs.append((gt_label, pred_label))

        return matched_pairs




class LocalizationMetrics:
    """Evaluate the detection of instance masks without looking at the class.
 
    Uses the same IoU matching as InstanceDetectionMetrics.
    """

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def evaluate(self, pred_data, gt_data):
        """Compute TP, FP, FN, precision, recall, F1 and the mean IoU of all TPs.
 
        Returns a dict with these values.
        """
        gt_dict = {item["id"]: item["annotations"] for item in gt_data.get("items", [])}

        total_tp = 0
        total_fp = 0
        total_fn = 0
        matched_ious = []

        for pred_item in pred_data.get("items", []):
            image_id = pred_item.get("id")
            if image_id not in gt_dict:
                continue

            pred_anns = [a for a in pred_item.get("annotations", []) if a.get("type") == "mask"]
            gt_anns = [a for a in gt_dict[image_id] if a.get("type") == "mask"]

            if not pred_anns and gt_anns:
                total_fn += len(gt_anns)
                continue

            if pred_anns and not gt_anns:
                total_fp += len(pred_anns)
                continue

            if not pred_anns and not gt_anns:
                continue

            pred_rles = [p["rle"] for p in pred_anns]
            gt_rles = [g["rle"] for g in gt_anns]
            iou_matrix = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles))

            matched_gts = set()
            matched_preds = set()

            flat_indices = np.argsort(iou_matrix, axis=None)[::-1]

            for idx in flat_indices:
                p_idx, g_idx = np.unravel_index(idx, iou_matrix.shape)

                if p_idx in matched_preds or g_idx in matched_gts:
                    continue

                iou_val = iou_matrix[p_idx, g_idx]
                if iou_val < self.iou_threshold:
                    break

                matched_preds.add(p_idx)
                matched_gts.add(g_idx)
                total_tp += 1
                matched_ious.append(iou_val)

            total_fp += (len(pred_anns) - len(matched_preds))
            total_fn += (len(gt_anns) - len(matched_gts))

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        loc_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

        return {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": loc_f1,
            "mean_iou": mean_iou,
        }