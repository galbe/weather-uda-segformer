"""
Standalone evaluation script.

Evaluate one or two checkpoints on ACDC val, Cityscapes val, or both.

Usage — single checkpoint:
    python evaluate.py --checkpoint outputs/step1_gta2cs/best.pth

Usage — before/after comparison:
    python evaluate.py \\
        --before outputs/step1_gta2cs/best.pth \\
        --after  outputs/step2_cs2acdc/best.pth

Flags:
    --dataset    acdc (default) | cityscapes | both
    --split      val (default) | train
    --conditions fog rain snow night  (ACDC only, default: all four)
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from datasets.acdc import ACDCDataset, ALL_CONDITIONS
from datasets.cityscapes import CityscapesDataset
from datasets.label_utils import TRAINID_TO_NAME
from utils.metrics import (
    make_confusion_matrix, update_confusion_matrix,
    compute_iou, format_iou_table,
)

NUM_CLASSES   = 19
ACDC_ROOT     = os.path.expanduser("~/data/acdc")
CS_ROOT       = os.path.expanduser("~/data/cityscapes")


def load_model(ckpt_path: str, device, backbone: str = "nvidia/mit-b2") -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        backbone,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate_acdc(model, loader, conditions, device):
    cond_conf = {c: make_confusion_matrix() for c in conditions}
    overall   = make_confusion_matrix()

    for batch in tqdm(loader, desc="  running", leave=False, file=sys.stdout):
        imgs   = batch["image"].to(device)
        labels = batch["label"].numpy()
        conds  = batch["condition"]
        B, _, H, W = imgs.shape

        logits = model(pixel_values=imgs).logits
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        preds  = logits.argmax(1).cpu().numpy()

        for pred, label, cond in zip(preds, labels, conds):
            update_confusion_matrix(overall, pred, label)
            update_confusion_matrix(cond_conf[cond], pred, label)

    iou_all, miou_all = compute_iou(overall)
    per_cond = {c: compute_iou(cond_conf[c])[1] for c in conditions}
    return {"miou": miou_all, "per_class": iou_all, "per_cond": per_cond}


@torch.no_grad()
def evaluate_cs(model, loader, device):
    overall = make_confusion_matrix()

    for batch in tqdm(loader, desc="  running", leave=False, file=sys.stdout):
        imgs   = batch["image"].to(device)
        labels = batch["label"].numpy()
        B, _, H, W = imgs.shape

        logits = model(pixel_values=imgs).logits
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        preds  = logits.argmax(1).cpu().numpy()

        for pred, label in zip(preds, labels):
            update_confusion_matrix(overall, pred, label)

    iou_all, miou_all = compute_iou(overall)
    return {"miou": miou_all, "per_class": iou_all}


def print_acdc_result(label: str, result: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(format_iou_table(result["per_class"], result["miou"]))
    print()
    print("  Per-condition mIoU:")
    for cond, v in result["per_cond"].items():
        print(f"    {cond:<8s}  {v*100:.2f}%")


def print_cs_result(label: str, result: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(format_iou_table(result["per_class"], result["miou"]))


def print_acdc_comparison(before: dict, after: dict, conditions):
    print(f"\n{'='*60}")
    print("  ACDC COMPARISON  (before → after second UDA step)")
    print(f"{'='*60}")
    delta = (after["miou"] - before["miou"]) * 100
    print(f"  Overall mIoU:  {before['miou']*100:.2f}%  →  {after['miou']*100:.2f}%  (Δ {delta:+.2f}%)")
    print()
    for cond in conditions:
        b = before["per_cond"][cond] * 100
        a = after["per_cond"][cond]  * 100
        print(f"  {cond:<8s}  {b:.2f}%  →  {a:.2f}%  (Δ {a-b:+.2f}%)")


def print_cs_comparison(before: dict, after: dict):
    print(f"\n{'='*60}")
    print("  CITYSCAPES COMPARISON  (before → after second UDA step)")
    print(f"{'='*60}")
    delta = (after["miou"] - before["miou"]) * 100
    print(f"  Overall mIoU:  {before['miou']*100:.2f}%  →  {after['miou']*100:.2f}%  (Δ {delta:+.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--before",     default=None)
    parser.add_argument("--after",      default=None)
    parser.add_argument("--dataset",    default="acdc",
                        choices=["acdc", "cityscapes", "both"])
    parser.add_argument("--split",      default="val",
                        choices=["val", "train"])
    parser.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS))
    parser.add_argument("--backbone",   default="nvidia/mit-b2",
                        help="HuggingFace model ID, e.g. nvidia/mit-b5")
    args = parser.parse_args()

    if not (args.checkpoint or (args.before and args.after)):
        parser.error("Provide --checkpoint, or both --before and --after")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    run_acdc = args.dataset in ("acdc", "both")
    run_cs   = args.dataset in ("cityscapes", "both")

    acdc_loader = cs_loader = None
    if run_acdc:
        ds_acdc = ACDCDataset(ACDC_ROOT, split=args.split,
                              conditions=args.conditions,
                              crop_size=None, labeled=True)
        acdc_loader = DataLoader(ds_acdc, batch_size=1, shuffle=False,
                                 num_workers=2, pin_memory=True)
        print(f"ACDC        : {len(ds_acdc)} images ({args.split}, {args.conditions})")
    if run_cs:
        ds_cs = CityscapesDataset(CS_ROOT, split=args.split,
                                  crop_size=None, labeled=True)
        cs_loader = DataLoader(ds_cs, batch_size=1, shuffle=False,
                               num_workers=2, pin_memory=True)
        print(f"Cityscapes  : {len(ds_cs)} images ({args.split})")

    if args.checkpoint:
        model = load_model(args.checkpoint, device, args.backbone)
        if run_acdc:
            r = evaluate_acdc(model, acdc_loader, args.conditions, device)
            print_acdc_result(f"{args.checkpoint}  [ACDC]", r)
        if run_cs:
            r = evaluate_cs(model, cs_loader, device)
            print_cs_result(f"{args.checkpoint}  [Cityscapes]", r)

    else:
        print(f"\nLoading BEFORE checkpoint: {args.before}")
        m_before = load_model(args.before, device, args.backbone)
        acdc_before = evaluate_acdc(m_before, acdc_loader, args.conditions, device) if run_acdc else None
        cs_before   = evaluate_cs(m_before, cs_loader, device)                       if run_cs   else None
        del m_before
        torch.cuda.empty_cache()

        print(f"\nLoading AFTER checkpoint:  {args.after}")
        m_after = load_model(args.after, device, args.backbone)
        acdc_after = evaluate_acdc(m_after, acdc_loader, args.conditions, device) if run_acdc else None
        cs_after   = evaluate_cs(m_after, cs_loader, device)                       if run_cs   else None
        del m_after

        if run_acdc:
            print_acdc_result("BEFORE (GTA5→CS only)  [ACDC val]",    acdc_before)
            print_acdc_result("AFTER  (GTA5→CS→ACDC)  [ACDC val]",    acdc_after)
            print_acdc_comparison(acdc_before, acdc_after, args.conditions)
        if run_cs:
            print_cs_result("BEFORE (GTA5→CS only)  [Cityscapes val]", cs_before)
            print_cs_result("AFTER  (GTA5→CS→ACDC)  [Cityscapes val]", cs_after)
            print_cs_comparison(cs_before, cs_after)


if __name__ == "__main__":
    main()
