"""Loss functions for all MTL tasks.
 
All losses are LibMTL losses (AbsLoss), so LibMTL can record their values per
epoch. get_loss_function chooses the loss of a task with the loss_function name in
task_definition of the config.
 
The pixel-wise losses only use valid pixels: pixels with a GT value of min_valid
(-1.5) or lower are ignored. These are the padded pixels and dropout holes, which
have the value -2.0 (see bbch_dataset.py).
"""


from LibMTL.loss import AbsLoss
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedAbsLoss(AbsLoss):
    """AbsLoss that records the loss value as a detached float."""
    def _update_loss(self, pred, gt):
        """Compute the loss and record its value and the batch size.
 
        Returns the loss tensor (with gradient).
        """
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.detach().item())  # type: ignore
        self.bs.append(pred.size()[0])
        return loss


class CELoss(FixedAbsLoss):
    """Cross-entropy loss for classification."""
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_loss(self, pred, gt):
        """Compute the cross-entropy of the logits pred and the class indices gt.
 
        Returns the mean loss.
        """
        loss = self.loss_fn(pred, gt)
        return loss


class FocalLoss(FixedAbsLoss, nn.Module):
    """Focal loss for classification.
 
    Down-weights easy samples with the factor (1 - p)^gamma, so the training
    focuses on hard samples.
    """
    def __init__(self, gamma=2.0):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.gamma = gamma
        
    def compute_loss(self, pred, gt):  # type: ignore
        """Compute the focal loss of the logits pred and the class indices gt.
 
        Returns the mean loss.
        """
        ce_loss = F.cross_entropy(pred, gt, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class MSELoss(FixedAbsLoss, nn.Module):
    """Mean squared error for the BBCH regression."""
    def __init__(self):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.loss_fn = nn.MSELoss()
        
    def compute_loss(self, pred, gt):
        """Compute the loss. Returns the mean loss."""
        return self.loss_fn(pred, gt)


class L1Loss(FixedAbsLoss, nn.Module):
    """Mean absolute error for the BBCH regression."""
    def __init__(self):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.loss_fn = nn.L1Loss()
        
    def compute_loss(self, pred, gt):
        """Compute the loss. Returns the mean loss."""
        return self.loss_fn(pred, gt)


class HuberLoss(FixedAbsLoss, nn.Module):
    """Huber loss for the BBCH regression (quadratic below delta, linear above)."""
    def __init__(self, delta=0.5):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.loss_fn = nn.HuberLoss(delta=delta)
        
    def compute_loss(self, pred, gt):
        """Compute the loss. Returns the mean loss."""
        return self.loss_fn(pred, gt)


class PPRMSELoss(FixedAbsLoss, nn.Module):
    """Mean squared error for the ppr_cm regression."""
    def __init__(self):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.loss_fn = nn.MSELoss()
        
    def compute_loss(self, pred, gt):  # type: ignore
        """Compute the loss. Returns the mean loss."""
        return self.loss_fn(pred, gt)


class PPRL1Loss(FixedAbsLoss, nn.Module):
    """Mean absolute error for the ppr_cm regression."""
    def __init__(self):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.loss_fn = nn.L1Loss()
        
    def compute_loss(self, pred, gt):  # type: ignore
        """Compute the loss. Returns the mean loss."""
        return self.loss_fn(pred, gt)


class PPRHuberLoss(FixedAbsLoss, nn.Module):
    """Huber loss for the ppr_cm regression (quadratic below delta, linear above)."""
    def __init__(self, delta=0.5):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.loss_fn = nn.HuberLoss(delta=delta)
        
    def compute_loss(self, pred, gt):  # type: ignore
        """Compute the loss. Returns the mean loss."""
        return self.loss_fn(pred, gt)


class SegmentationCELoss(FixedAbsLoss, nn.Module):
    """Pixel-wise cross-entropy for the semantic segmentation.
 
    Pixels with ignore_index (255) are ignored. Optional class weights can be given.
    """
    def __init__(self, ignore_index=255, weight=None):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer('weight', weight.detach().clone().float())
        else:
            self.weight = None

    def compute_loss(self, pred, gt):  # type: ignore
        """Compute the cross-entropy of the logits pred (B x C x H x W) and gt.
 
        Returns the mean loss over all pixels that are not ignored.
        """
        if self.weight is not None and self.weight.device != pred.device:
            self.weight = self.weight.to(pred.device)

        return F.cross_entropy(
            pred, 
            gt, 
            weight=self.weight, 
            ignore_index=self.ignore_index
        )


class WeightedSegmentationCELoss(FixedAbsLoss, nn.Module):
    """Pixel-wise cross-entropy with fixed class weights.
 
    The default weights [1, 2, 2, 5] (background, weed, neighbor, main_plant) give
    the small classes, above all the main plant, more weight.
    """
    def __init__(self, ignore_index=255, custom_weights=[1.0, 2.0, 2.0, 5.0]):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.ignore_index = ignore_index
        weight_tensor = torch.tensor(custom_weights).float()
        self.register_buffer('weight', weight_tensor)

    def compute_loss(self, pred, gt):  # type: ignore
        """Compute the weighted cross-entropy of the logits pred and gt.
 
        Returns the weighted mean loss over all pixels that are not ignored.
        """
        if self.weight.device != pred.device:
            self.weight = self.weight.to(pred.device)
 
        return F.cross_entropy(
            pred, 
            gt, 
            weight=self.weight, 
            ignore_index=self.ignore_index
        )


class MaskedMSELoss(FixedAbsLoss, nn.Module):
    """Mean squared error over the valid pixels (GT > min_valid)."""
    def __init__(self, min_valid=-1.5):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.min_valid = min_valid

    def compute_loss(self, pred, gt):
        """Compute the loss. Returns the mean loss over the valid pixels."""
        mask = (gt > self.min_valid).float()
        mse = F.mse_loss(pred, gt, reduction='none')
        return (mse * mask).sum() / (mask.sum() + 1e-8)


class MaskedL1Loss(FixedAbsLoss, nn.Module):
    """Mean absolute error over the valid pixels (GT > min_valid)."""
    def __init__(self, min_valid=-1.5):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.min_valid = min_valid

    def compute_loss(self, pred, gt):
        """Compute the loss. Returns the mean loss over the valid pixels."""
        mask = (gt > self.min_valid).float()
        l1 = F.l1_loss(pred, gt, reduction='none')
        return (l1 * mask).sum() / (mask.sum() + 1e-8)


class MaskedHuberLoss(FixedAbsLoss, nn.Module):
    """Huber loss over the valid pixels (GT > min_valid)."""
    def __init__(self, min_valid=-1.5, delta=0.5):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.min_valid = min_valid
        self.delta = delta

    def compute_loss(self, pred, gt):
        """Compute the loss. Returns the mean loss over the valid pixels."""
        mask = (gt > self.min_valid).float()
        huber = F.huber_loss(pred, gt, reduction='none', delta=self.delta)
        return (huber * mask).sum() / (mask.sum() + 1e-8)


class DistanceWeightedHuberLoss(FixedAbsLoss, nn.Module):
    """Huber loss over the valid pixels with a higher weight for plant pixels.
 
    Plant pixels (GT >= 0, the real distances of the main plant) get plant_weight,
    all other valid pixels get 1.
    """
    def __init__(self, min_valid=-1.5, plant_weight=5.0, delta=0.5):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.min_valid = min_valid
        self.plant_weight = plant_weight
        self.delta = delta

    def compute_loss(self, pred, gt):
        """Compute the loss.
 
        Returns the weighted mean loss over the valid pixels, or 0 if no pixel is valid.
        """
        mask = gt > self.min_valid
        if not mask.any():
            return pred.sum() * 0
        huber_loss = F.huber_loss(pred, gt, reduction='none', delta=self.delta)        
        weights = torch.ones_like(gt)
        weights[gt >= 0] = self.plant_weight     
        valid_weighted_loss = (huber_loss * weights) * mask.float()
        
        return valid_weighted_loss.sum() / (mask.float() * weights).sum()


class DistanceWeightedMSELoss(FixedAbsLoss, nn.Module):
    """Mean squared error over the valid pixels with a higher weight for plant pixels.
 
    Plant pixels (GT >= 0) get plant_weight, all other valid pixels get 1.
    """
    def __init__(self, min_valid=-1.5, plant_weight=5.0):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.min_valid = min_valid
        self.plant_weight = plant_weight

    def compute_loss(self, pred, gt):
        """Compute the loss. Returns the weighted mean loss over the valid pixels."""
        mask = (gt > self.min_valid).float()
        mse = F.mse_loss(pred, gt, reduction='none')
        
        weights = torch.ones_like(gt)
        weights[gt >= 0] = self.plant_weight
        
        weighted_mse = mse * weights * mask
        return weighted_mse.sum() / ((mask * weights).sum() + 1e-8)


class DistanceWeightedL1Loss(FixedAbsLoss, nn.Module):
    """Mean absolute error over the valid pixels with a higher weight for plant pixels.
 
    Plant pixels (GT >= 0) get plant_weight, all other valid pixels get 1.
    """
    def __init__(self, min_valid=-1.5, plant_weight=5.0):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.min_valid = min_valid
        self.plant_weight = plant_weight

    def compute_loss(self, pred, gt):
        """Compute the loss. Returns the weighted mean loss over the valid pixels."""
        mask = (gt > self.min_valid).float()
        l1 = F.l1_loss(pred, gt, reduction='none')
        
        weights = torch.ones_like(gt)
        weights[gt >= 0] = self.plant_weight
        
        weighted_l1 = l1 * weights * mask
        return weighted_l1.sum() / ((mask * weights).sum() + 1e-8)


class DirectionCosineSimilarityLoss(FixedAbsLoss, nn.Module):
    """1 - cosine similarity between predicted and GT direction vectors.
 
    Only valid pixels with a GT vector that is not [0, 0] are used, so the
    background is ignored.
    """
    def __init__(self, min_valid=-1.5):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.min_valid = min_valid

    def compute_loss(self, pred, gt):
        """Compute the loss. Returns the mean loss over the used pixels."""
        valid_mask = (gt[:, 0:1, :, :] > self.min_valid).float()
        not_zero_mask = (gt.abs().sum(dim=1, keepdim=True) > 1e-4).float()
        mask = valid_mask * not_zero_mask
        
        cos_sim = F.cosine_similarity(pred, gt, dim=1).unsqueeze(1)
        loss = (1.0 - cos_sim) * mask
        return loss.sum() / (mask.sum() + 1e-8)


class DirectionMSELoss(FixedAbsLoss, nn.Module):
    """Mean squared error of the direction vectors over the valid pixels.
 
    Unlike the cosine losses, background pixels (GT [0, 0]) are also used.
    """
    def __init__(self, min_valid=-1.5):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.min_valid = min_valid

    def compute_loss(self, pred, gt):
        """Compute the loss. Returns the summed squared error per valid pixel."""
        mask = (gt[:, 0:1, :, :] > self.min_valid).float()
        mse = F.mse_loss(pred, gt, reduction='none')
        return (mse * mask).sum() / (mask.sum() + 1e-8)


class DirectionHuberCosineLoss(FixedAbsLoss, nn.Module):
    """Combination of a cosine loss and a Huber loss for the direction vectors.
 
    The cosine part (2 * (1 - cosine similarity)) penalizes the angle error, the
    Huber part (weighted with weight_huber) the difference of the vector values.
    Only valid pixels with a GT vector that is not [0, 0] are used.
    """
    def __init__(self, min_valid=-1.5, weight_huber=1.0, delta=0.5):
        AbsLoss.__init__(self)
        nn.Module.__init__(self)
        self.min_valid = min_valid
        self.weight_huber = weight_huber
        self.huber = nn.HuberLoss(reduction='none', delta=delta)

    def compute_loss(self, pred, gt):
        """Compute the loss.
 
        Returns the mean loss over the used pixels, or 0 if no pixel is used.
        """
        valid_mask = (gt[:, 0:1, :, :] > self.min_valid).float()
        not_zero_mask = (gt.abs().sum(dim=1, keepdim=True) > 1e-4).float()
        mask = valid_mask * not_zero_mask
        
        if mask.sum() < 1:
            return pred.sum() * 0.0

        cos_sim = F.cosine_similarity(pred, gt, dim=1).unsqueeze(1)
        loss_cos = (1.0 - cos_sim) * mask * 2
        loss_huber = self.huber(pred, gt) * mask

        total_loss = (loss_cos.sum() + self.weight_huber * loss_huber.sum()) / (mask.sum() + 1e-8)
        return total_loss


def get_loss_function(task_name: str, params):
    """Create the loss function of a task from params.loss_function.
 
    Supported names per task:
    - bbch_classification, damaged_classification: ce, focal
    - bbch_regression, pprcm_regression: mse, l1, huber
    - semantic_segmentation: ce, weighted_ce
    - distance_regression: mse, l1, huber, weighted_huber, weighted_l1, weighted_mse
    - direction_regression: mse, cos_sim, huber_cos
    Raises ValueError for an unsupported name.
 
    Returns the loss object.
    """
    loss_function_name = params.loss_function
    
    if task_name == "bbch_classification":
        if loss_function_name == "ce":
            return CELoss()
        elif loss_function_name == "focal":
            return FocalLoss(gamma=2.0)
        else:
            raise ValueError(f"Classification Loss {loss_function_name} not supported.")
    
    elif task_name == "bbch_regression":
        if loss_function_name == "mse":
            return MSELoss()
        elif loss_function_name == "l1":
            return L1Loss()
        elif loss_function_name == "huber":
            return HuberLoss(delta=0.5)
        else:
            raise ValueError(f"Regression Loss {loss_function_name} not supported.")
    
    elif task_name == "pprcm_regression":
        if loss_function_name == "mse":
            return PPRMSELoss()
        elif loss_function_name == "l1":
            return PPRL1Loss()
        elif loss_function_name == "huber":
            return PPRHuberLoss(delta=0.5) 
        else:
            raise ValueError(f"PPRCM Loss {loss_function_name} not supported.")
    
    elif task_name == "damaged_classification":
        if loss_function_name == "ce":
            return CELoss()
        elif loss_function_name == "focal":
            return FocalLoss(gamma=2.0)
        else:
            raise ValueError(f"Damage Classification Loss {loss_function_name} not supported.")
    
    elif task_name == "semantic_segmentation":
        if loss_function_name == "ce":
            return SegmentationCELoss(ignore_index=255, weight=None)
        elif loss_function_name == "weighted_ce":
            return WeightedSegmentationCELoss(ignore_index=255, custom_weights=[1.0, 2.0, 2.0, 5.0])
        else:
            raise ValueError(f"Segmentation Loss {loss_function_name} not supported.")

    elif task_name == "distance_regression":
        if loss_function_name == "mse":
            return MaskedMSELoss()
        elif loss_function_name == "l1":
            return MaskedL1Loss()
        elif loss_function_name == "huber":
            return MaskedHuberLoss()
        elif loss_function_name == "weighted_huber":
            return DistanceWeightedHuberLoss(plant_weight=5.0, delta=0.5)
        elif loss_function_name == "weighted_l1":
            return DistanceWeightedL1Loss(plant_weight=5.0)
        elif loss_function_name == "weighted_mse":
            return DistanceWeightedMSELoss(plant_weight=5.0)
        else:
            raise ValueError(f"Distance Loss {loss_function_name} not supported.")

    elif task_name == "direction_regression":
        if loss_function_name == "mse":
            return DirectionMSELoss()
        elif loss_function_name == "cos_sim":
            return DirectionCosineSimilarityLoss()
        elif loss_function_name == "huber_cos":
            return DirectionHuberCosineLoss(delta=0.5)
        else:
            raise ValueError(f"Direction Loss {loss_function_name} not supported.")