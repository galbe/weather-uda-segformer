"""
Cityscapes dataset.
- labeled=True, crop_size set  → training source (step 2) or evaluation
- labeled=False, crop_size set → unlabeled target (step 1 UDA)
- labeled=True, crop_size=None → evaluation at (512, 1024)
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
_EVAL_H, _EVAL_W = 512, 1024


class CityscapesDataset(Dataset):
    def __init__(self, root: str, split: str = "train",
                 crop_size: int = None, labeled: bool = True):
        self.labeled = labeled
        self.lut = build_cs_lut()

        img_dir = Path(root) / "leftImg8bit" / split
        lbl_dir = Path(root) / "gtFine" / split

        self.pairs = []
        for img_p in sorted(img_dir.rglob("*_leftImg8bit.png")):
            city = img_p.parent.name
            stem = img_p.stem.replace("_leftImg8bit", "")
            lbl_p = lbl_dir / city / f"{stem}_gtFine_labelIds.png"
            if labeled and not lbl_p.exists():
                continue
            self.pairs.append((img_p, lbl_p if labeled else None))

        assert len(self.pairs) > 0, f"No Cityscapes images found in {img_dir}"

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
        img_p, lbl_p = self.pairs[idx]
        image = np.array(Image.open(img_p).convert("RGB"))
        dummy = np.zeros(image.shape[:2], dtype=np.uint8)
        label = self.lut[np.array(Image.open(lbl_p)).astype(np.uint8)] if self.labeled else dummy

        r = self._aug(image=image, mask=label)
        img_t = torch.from_numpy(r["image"].transpose(2, 0, 1).copy())
        lbl_t = torch.from_numpy(r["mask"].astype(np.int64))

        out = {"image": img_t}
        if self.labeled:
            out["label"] = lbl_t
        return out
