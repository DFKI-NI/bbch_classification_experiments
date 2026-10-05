"""Albumentations pipelines for the MTL training and evaluation."""


import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_transforms(params, is_training=True):
    """Build the albumentations pipeline for training or evaluation.
 
    In training, color, noise and blur augmentations, random dropout holes and a
    random shift (and scale in ratio mode) are applied. Flips and rotations are not
    done here but in BBCH_Dataset, because they also change the direction vectors.
 
    Two resize modes (params.img_resize):
    - ratio: scale the longest side to img_size and pad to img_size x img_size.
    - gsd: only pad to img_size x img_size. The dataset already scaled all images
      with the same factor, so the GSD stays the same for all images.
 
    Padded pixels and dropout holes get the value -999.0 in all masks, so the
    dataset can mark them as ignored. Raises ValueError for an unknown resize mode.
 
    Returns the A.Compose pipeline.
    """
    size = params.img_size
    mask_fill_value = -999.0

    aug_ops = []
    if is_training:
        aug_ops += [
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=10, p=0.4),
            A.RGBShift(p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0, 0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.Blur(p=1.0),
                A.MotionBlur(p=1.0),
            ], p=0.3),
            A.CoarseDropout(
                num_holes_range=(4, 10),
                hole_height_range=(0.02, 0.08),
                hole_width_range=(0.02, 0.08),
                fill=0,
                fill_mask=mask_fill_value,
                p=0.3
            )
        ]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if params.img_resize == "ratio":
        resize_ops = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=mask_fill_value
            ),
        ]
        if is_training:
            resize_ops.append(
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=mask_fill_value,
                    p=0.5
                )
            )
    elif params.img_resize == "gsd":
        resize_ops = [
            A.PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=mask_fill_value
            ),
        ]
        if is_training:
            resize_ops.append(
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=mask_fill_value,
                    p=0.5
                )
            )
    else:
        raise ValueError(f"Invalid img_resize mode '{params.img_resize}'. Expected 'ratio' or 'gsd'.")

    additional_targets = {
        "dist_mask": "mask",
        "dir_mask": "mask"
    }

    return A.Compose(
        [
            *aug_ops,
            *resize_ops,
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ],
        additional_targets=additional_targets,
        is_check_shapes=False,
        seed=getattr(params, "seed", 42)
    )