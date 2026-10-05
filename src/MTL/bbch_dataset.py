"""Dataset and DataLoaders for the MTL training on the cropped maize plants.
 
Each item is one cropped plant from the preprocessing (MTL/images and
MTL/annotations). The dataset loads the image and the masks stored in the item
attributes, applies the augmentations and returns the image together with the
targets of all 7 tasks.
 
Ignored pixels (padding and dropout holes) get special values in the targets:
255 in the semantic mask and -2.0 in the distance and direction masks.
"""


import os
import json
import random
import base64
import zlib
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from augmentations import get_transforms


def seed_worker(worker_id):                                                      
    """Give each DataLoader worker its own random state in each epoch.
 
    Without this, all workers could produce the same augmentations.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    seed = torch.initial_seed() % (2 ** 32) 
    random.seed(seed)
    np.random.seed(seed)
    transform = getattr(info.dataset, "transform", None)
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)
    else:
        print("[WARN] transform has no set_random_seed.")


def get_dataloaders(params):
    """Create the DataLoaders for train, val and test.
 
    Only the train set is augmented and shuffled.
 
    Returns (train_loader, val_loader, test_loader).
    """
    train_transform = get_transforms(params, is_training=True)
    val_transform = get_transforms(params, is_training=False)

    images_path = os.path.join(params.dataset_path, "images")
    annotations_path = os.path.join(params.dataset_path, "annotations")

    train_dataset = BBCH_Dataset(
        ann_path=os.path.join(annotations_path, "train.json"),
        img_dir=os.path.join(images_path, "train"),
        params=params,
        transform=train_transform,
        is_training=True                     
    )

    val_dataset = BBCH_Dataset(
        ann_path=os.path.join(annotations_path, "val.json"),
        img_dir=os.path.join(images_path, "val"),
        params=params,
        transform=val_transform,
        is_training=False                    
    )

    test_dataset = BBCH_Dataset(
        ann_path=os.path.join(annotations_path, "test.json"),
        img_dir=os.path.join(images_path, "test"),
        params=params,
        transform=val_transform,
        is_training=False                    
    )

    print(f"Dataset subset sizes | train {len(train_dataset)} | "
          f"val {len(val_dataset)} | test {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.train_bs,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker          
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def decode_base64_mask(mask_dict: dict) -> np.ndarray:
    """Decode a float16 mask (distance or direction) stored as zlib+base64.
 
    Returns the mask as a float32 array.
    """
    shape = mask_dict["shape"]
    b64_string = mask_dict["data"]

    compressed_bytes = base64.b64decode(b64_string)
    raw_bytes = zlib.decompress(compressed_bytes)

    array = np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape)
    return array.astype(np.float32)


def decode_semantic_base64_mask(mask_dict: dict) -> np.ndarray:
    """Decode a uint8 semantic mask (classes 0 to 3) stored as zlib+base64.
 
    Returns the mask as a uint8 array.
    """
    shape = mask_dict["shape"]
    b64_string = mask_dict["data"]

    compressed_bytes = base64.b64decode(b64_string)
    raw_bytes = zlib.decompress(compressed_bytes)

    array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape)
    return array


def verify_global_scale(params) -> None:
    """Check that params.global_scale fits the largest image side of the dataset.
 
    In gsd mode, each image is scaled by img_size / global_scale. If an image side
    is larger than global_scale, the scaled image is larger than img_size, and a
    warning is printed.
    """
    annotations_path = os.path.join(params.dataset_path, "annotations")
    max_dim = 0
    for split in ("train", "val", "test"):
        path = os.path.join(annotations_path, f"{split}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("items", []):
            size = item.get("image", {}).get("size", [])
            if len(size) == 2:
                max_dim = max(max_dim, int(max(size)))

    configured = float(getattr(params, "global_scale", 0))
    print(f"global_scale: {configured:.0f} | biggest image border in dataset: {max_dim}")
    if max_dim > configured:
        print(f"  WARNING: Images are scaled to "
              f"{max_dim * params.img_size / configured:.0f} pixel, "
              f"but img_size is {params.img_size}.")


class BBCH_Dataset(Dataset):
    """Dataset of cropped maize plants with the targets of all MTL tasks.
 
    __getitem__ returns (image, labels). labels is a dict with one target per task:
    - bbch_classification: class index of the BBCH stage (0 to 7)
    - bbch_regression: BBCH stage scaled to 0..1 with bbch_min and bbch_max
    - pprcm_regression: ppr_cm clipped and scaled to 0..1 with ppr_min and ppr_max
    - damaged_classification: 1 if damage is visible, otherwise 0
    - semantic_segmentation: semantic mask (H x W), see step4_create_semantic_masks.py
    - distance_regression: distance mask (1 x H x W)
    - direction_regression: direction mask (2 x H x W)
 
    The plant-level labels are read from the main plant annotation
    (is_main_plant = True), or from the first annotation if no main plant is marked.
    """
        
    def __init__(self, ann_path, img_dir, params, transform=None, is_training=False):  
        self.params = params
        self.transform = transform
        self.img_dir = img_dir
        self.current_epoch = 0
        self.is_training = is_training                                                

        with open(ann_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.items = self.data["items"]

        self.bbch_map = getattr(params, 'bbch_map', {str(i): i - 10 for i in range(10, 18)})
        self.bbch_min = getattr(params, 'bbch_min')
        self.bbch_max = getattr(params, 'bbch_max')
        self.ppr_min = getattr(params, 'ppr_min')
        self.ppr_max = getattr(params, 'ppr_max')

    def set_epoch(self, epoch):
        """Set the current epoch.
 
        The epoch is used to seed the flips and rotations, so they change in each
        epoch.
        """
        self.current_epoch = epoch

    def __len__(self):
        """Return the number of items."""
        return len(self.items)

    def __getitem__(self, idx):
        """Load one item and return the image and the targets of all tasks.
 
        Steps: load the image and the masks, scale them in gsd mode, apply random
        flips and 90 degree rotations (only in training, with the matching sign
        correction of the direction vectors), apply the albumentations pipeline and
        mark the padded pixels as ignored. Raises FileNotFoundError if the image is
        missing and KeyError if the item has no annotation.
 
        Returns (image, labels), see the class docstring.
        """
        item = self.items[idx]
        item_id = item["id"]
        attr = item.get("attr", {})

        rel_img_path = item["image"].get("file_name", item["image"].get("path"))
        filename = os.path.basename(rel_img_path)
        img_path = os.path.join(self.img_dir, filename)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_h, img_w = item["image"]["size"]

        if "semantic_mask" in attr:
            sem_mask = decode_semantic_base64_mask(attr["semantic_mask"])
            if sem_mask.ndim == 3:
                sem_mask = np.squeeze(sem_mask, axis=-1)
        else:
            sem_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if "distance_mask" in attr:
            dist_mask = decode_base64_mask(attr["distance_mask"])
            if dist_mask.ndim == 3:
                dist_mask = np.squeeze(dist_mask, axis=-1)
        else:
            dist_mask = np.full((img_h, img_w), -1.0, dtype=np.float32)

        if "direction_mask" in attr:
            dir_mask_raw = decode_base64_mask(attr["direction_mask"])  # (2, H, W)
            dir_mask = np.ascontiguousarray(np.transpose(dir_mask_raw, (1, 2, 0)))
        else:
            dir_mask = np.zeros((img_h, img_w, 2), dtype=np.float32)

        sem_mask = sem_mask.astype(np.float32)

        if getattr(self.params, 'img_resize', None) == 'gsd':
            target_size = getattr(self.params, 'img_size', 448)
            max_dataset_dim = float(getattr(self.params, 'global_scale', 1022.0))
            scale_factor = target_size / max_dataset_dim

            new_w = int(image.shape[1] * scale_factor)
            new_h = int(image.shape[0] * scale_factor)

            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            sem_mask = cv2.resize(sem_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            dist_mask = cv2.resize(dist_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            dir_mask = cv2.resize(dir_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if self.is_training:                                                        
            state = random.getstate()
            np_state = np.random.get_state()
            random.seed(self.params.seed + idx + self.current_epoch)
            np.random.seed(self.params.seed + idx + self.current_epoch)

            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                sem_mask = cv2.flip(sem_mask, 1)
                dist_mask = cv2.flip(dist_mask, 1)
                dir_mask = cv2.flip(dir_mask, 1)
                dir_mask[:, :, 0] *= -1.0

            if random.random() < 0.5:
                image = cv2.flip(image, 0)
                sem_mask = cv2.flip(sem_mask, 0)
                dist_mask = cv2.flip(dist_mask, 0)
                dir_mask = cv2.flip(dir_mask, 0)
                dir_mask[:, :, 1] *= -1.0

            rotate_factor = random.randint(0, 3)
            if rotate_factor > 0:
                image = np.rot90(image, rotate_factor)
                sem_mask = np.rot90(sem_mask, rotate_factor)
                dist_mask = np.rot90(dist_mask, rotate_factor)
                dir_mask = np.rot90(dir_mask, rotate_factor)
                c, s = dir_mask[:, :, 0].copy(), dir_mask[:, :, 1].copy()
                if rotate_factor == 1:
                    dir_mask[:, :, 0], dir_mask[:, :, 1] = -s, c
                elif rotate_factor == 2:
                    dir_mask[:, :, 0], dir_mask[:, :, 1] = -c, -s
                elif rotate_factor == 3:
                    dir_mask[:, :, 0], dir_mask[:, :, 1] = s, -c

            random.setstate(state)
            np.random.set_state(np_state)

        if self.transform:
            transformed = self.transform(
                image=image,
                mask=sem_mask,
                dist_mask=dist_mask,
                dir_mask=dir_mask
            )
            image = transformed['image']
            sem_mask = transformed['mask']
            dist_mask = transformed['dist_mask']
            dir_mask = transformed['dir_mask']

            if isinstance(sem_mask, torch.Tensor):
                if sem_mask.ndim == 3 and sem_mask.shape[0] == 1:
                    sem_mask = sem_mask.squeeze(0)
            elif isinstance(sem_mask, np.ndarray):
                if sem_mask.ndim == 3 and sem_mask.shape[-1] == 1:
                    sem_mask = sem_mask.squeeze(-1)

            if isinstance(dist_mask, torch.Tensor):
                if dist_mask.ndim == 3 and dist_mask.shape[0] == 1:
                    dist_mask = dist_mask.squeeze(0)
            elif isinstance(dist_mask, np.ndarray):
                if dist_mask.ndim == 3 and dist_mask.shape[-1] == 1:
                    dist_mask = dist_mask.squeeze(-1)

            if isinstance(sem_mask, torch.Tensor):
                padding_mask = (sem_mask < -900.0)
                sem_mask_final = torch.round(sem_mask).long()
            else:
                padding_mask = torch.from_numpy(sem_mask < -900.0)
                sem_mask_final = torch.from_numpy(np.round(sem_mask)).long()

            sem_mask_final[padding_mask] = 255

            if isinstance(dist_mask, np.ndarray):
                dist_mask_final = torch.from_numpy(dist_mask.copy()).float()
            else:
                dist_mask_final = dist_mask.float()

            if isinstance(dir_mask, np.ndarray):
                dir_mask_tensor = torch.from_numpy(dir_mask.copy()).float()
                if dir_mask_tensor.ndim == 3 and dir_mask_tensor.shape[-1] == 2:
                    dir_mask_final = dir_mask_tensor.permute(2, 0, 1)
                else:
                    dir_mask_final = dir_mask_tensor
            else:
                if dir_mask.ndim == 3 and dir_mask.shape[-1] == 2:
                    dir_mask_final = dir_mask.permute(2, 0, 1).float()
                elif dir_mask.ndim == 3 and dir_mask.shape[0] == 2:
                    dir_mask_final = dir_mask.float()
                else:
                    dir_mask_final = dir_mask.float()

            dist_mask_final[padding_mask] = -2.0
            dir_mask_final[:, padding_mask] = -2.0

            invalid_indices = (sem_mask_final < 0) | (sem_mask_final > 3)
            sem_mask_final[invalid_indices & ~padding_mask] = 255

        else:
            sem_mask_final = torch.from_numpy(sem_mask.copy()).long()
            dist_mask_final = torch.from_numpy(dist_mask.copy()).float()
            dir_mask_final = torch.from_numpy(dir_mask.copy()).permute(2, 0, 1).float()

        main_ann = None
        for ann in item.get("annotations", []):
            attrs = ann.get("attributes", {})
            if attrs.get("is_main_plant") is True:
                main_ann = attrs
                break

        if main_ann is None and len(item.get("annotations", [])) > 0:
            main_ann = item["annotations"][0].get("attributes", {})

        if main_ann is None:
            raise KeyError(f"No attributes for main plant found in item {item_id}!")

        bbch_val = float(main_ann.get("bbch"))
        ppr_val = float(main_ann.get("ppr_cm"))
        is_damaged = int(main_ann.get("damage_visible"))

        bbch_idx = self.bbch_map.get(str(int(bbch_val)))
        bbch_norm = (bbch_val - self.bbch_min) / (self.bbch_max - self.bbch_min)
        ppr_clamped = max(self.ppr_min, min(ppr_val, self.ppr_max))
        ppr_norm = (ppr_clamped - self.ppr_min) / (self.ppr_max - self.ppr_min)

        labels = {
            'bbch_classification': torch.tensor(bbch_idx).long(),
            'bbch_regression': torch.tensor([bbch_norm]).float(),
            'pprcm_regression': torch.tensor([ppr_norm]).float(),
            'damaged_classification': torch.tensor(is_damaged).long(),
            'semantic_segmentation': sem_mask_final,
            'distance_regression': dist_mask_final.unsqueeze(0) if dist_mask_final.ndim == 2 else dist_mask_final,
            'direction_regression': dir_mask_final
        }

        return image, labels