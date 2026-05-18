"""Per-class IoU and mIoU from a confusion matrix."""
import numpy as np
from datasets.label_utils import TRAINID_TO_NAME, NUM_CLASSES, IGNORE_INDEX


def make_confusion_matrix(num_classes=NUM_CLASSES):
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def update_confusion_matrix(conf, pred: np.ndarray, label: np.ndarray):
    """
    pred, label: (H, W) numpy int arrays.
    Pixels where label == IGNORE_INDEX are skipped.
    """
    mask = (label != IGNORE_INDEX) & (label >= 0) & (label < NUM_CLASSES)
    p = pred[mask].astype(np.int64)
    l = label[mask].astype(np.int64)
    p = np.clip(p, 0, NUM_CLASSES - 1)
    np.add.at(conf, (l, p), 1)


def compute_iou(conf: np.ndarray):
    """Returns per-class IoU array and mean IoU (NaN classes excluded)."""
    inter = np.diag(conf)
    union = conf.sum(1) + conf.sum(0) - inter
    iou = np.where(union > 0, inter / union, np.nan)
    miou = float(np.nanmean(iou))
    return iou, miou


def format_iou_table(iou_per_class: np.ndarray, miou: float, prefix="") -> str:
    lines = [f"{prefix}mIoU: {miou*100:.2f}%"]
    for i, v in enumerate(iou_per_class):
        name = TRAINID_TO_NAME.get(i, f"class{i}")
        val  = f"{v*100:.1f}%" if not np.isnan(v) else "  N/A"
        lines.append(f"  {i:2d} {name:<16s} {val}")
    return "\n".join(lines)
