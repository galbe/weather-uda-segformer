"""GTA5 dataset — labeled source for UDA step 1."""
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_utils import build_gta5_lut

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


class GTA5Dataset(Dataset):
    """
    Only yields pairs where both image and label exist.
    (images 05001–07500 are skipped — labels missing from zip 02)
    """

    def __init__(self, root: str, crop_size: int = 512):
        images_dir = Path(root) / "images"
        labels_dir = Path(root) / "labels"

        label_stems = {p.stem for p in labels_dir.glob("*.png")}
        all_pairs = sorted([
            (images_dir / f"{s}.png", labels_dir / f"{s}.png")
            for s in label_stems
            if (images_dir / f"{s}.png").exists()
        ])
        # skip pairs where image and label have different spatial dimensions
        self.pairs = [
            (ip, lp) for ip, lp in all_pairs
            if Image.open(ip).size == Image.open(lp).size
        ]
        assert len(self.pairs) > 0, f"No GTA5 pairs found under {root}"

        self.lut = build_gta5_lut()
        self._aug = A.Compose([
            A.RandomCrop(height=crop_size, width=crop_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5),
            A.Normalize(mean=_MEAN, std=_STD),
        ], is_check_shapes=False)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, lbl_p = self.pairs[idx]
        image = np.array(Image.open(img_p).convert("RGB"))
        label = self.lut[np.array(Image.open(lbl_p)).astype(np.uint8)]

        r = self._aug(image=image, mask=label)
        img_t = torch.from_numpy(r["image"].transpose(2, 0, 1).copy())
        lbl_t = torch.from_numpy(r["mask"].astype(np.int64))
        return {"image": img_t, "label": lbl_t}
