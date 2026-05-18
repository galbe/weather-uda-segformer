"""
ACDC dataset.
- labeled=False, crop_size set → unlabeled target for UDA step 2 training
- labeled=True,  crop_size=None → evaluation at (540, 960)
"""
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_utils import build_cs_lut

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)
_EVAL_H, _EVAL_W = 540, 960
ALL_CONDITIONS = ("fog", "rain", "snow", "night")


class ACDCDataset(Dataset):
    def __init__(self, root: str, split: str = "val",
                 conditions=None, crop_size: int = None, labeled: bool = True):
        self.labeled = labeled
        self.lut = build_cs_lut()
        if conditions is None:
            conditions = ALL_CONDITIONS

        rgb_root = Path(root) / "rgb_anon"
        gt_root  = Path(root) / "gt"

        self.pairs = []
        for cond in conditions:
            for img_p in sorted((rgb_root / cond / split).rglob("*_rgb_anon.png")):
                lbl_name = img_p.name.replace("_rgb_anon.png", "_gt_labelIds.png")
                lbl_p    = gt_root / cond / split / img_p.parent.name / lbl_name
                if labeled and not lbl_p.exists():
                    continue
                self.pairs.append((img_p, lbl_p if labeled else None, cond))

        assert len(self.pairs) > 0, f"No ACDC images found under {root}/{split}"

        if crop_size:
            self._aug = A.Compose([
                A.RandomCrop(height=crop_size, width=crop_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5),
                A.Normalize(mean=_MEAN, std=_STD),
            ])
            self._eval_mode = False
        else:
            self._aug = A.Compose([
                A.Resize(height=_EVAL_H, width=_EVAL_W),
                A.Normalize(mean=_MEAN, std=_STD),
            ])
            self._eval_mode = True

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, lbl_p, cond = self.pairs[idx]
        image = np.array(Image.open(img_p).convert("RGB"))
        dummy = np.zeros(image.shape[:2], dtype=np.uint8)
        label = self.lut[np.array(Image.open(lbl_p)).astype(np.uint8)] if self.labeled else dummy

        r = self._aug(image=image, mask=label)
        img_t = torch.from_numpy(r["image"].transpose(2, 0, 1).copy())
        lbl_t = torch.from_numpy(r["mask"].astype(np.int64))

        out = {"image": img_t, "condition": cond}
        if self.labeled:
            out["label"] = lbl_t
        return out
