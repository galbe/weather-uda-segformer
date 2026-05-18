"""
Two-step UDA training script (Mean Teacher pseudo-label approach).

Step 1  GTA5 (labeled) → Cityscapes (unlabeled target):
    python train_uda.py --config configs/uda_gta2cs.yaml

Step 2  Cityscapes (labeled) → ACDC (unlabeled target),
        initialized from step-1 checkpoint:
    python train_uda.py --config configs/uda_cs2acdc.yaml \
                        --resume outputs/step1_gta2cs/best.pth

Smoke test (50 iters, no eval/save):
    python train_uda.py --config configs/uda_gta2cs.yaml --smoke

Auto-resume: if {out_dir}/resume.pth exists it is loaded automatically
             (model + teacher + optimizer + scheduler + step + best_miou).
"""
import argparse
import copy
import os
import random
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from datasets.gta5 import GTA5Dataset
from datasets.cityscapes import CityscapesDataset
from datasets.acdc import ACDCDataset
from utils.metrics import (
    make_confusion_matrix, update_confusion_matrix,
    compute_iou, format_iou_table,
)

NUM_CLASSES  = 19
IGNORE_INDEX = 255


# ── helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, decay: float):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)
    for t, s in zip(teacher.buffers(), student.buffers()):
        t.data.copy_(s.data)


def get_logits(model, pixel_values, target_hw):
    """Forward pass → logits upsampled to target_hw (H, W)."""
    logits = model(pixel_values=pixel_values).logits   # (B, C, H/4, W/4)
    return F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)


def src_ce_loss(logits, labels):
    return F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)


def pseudo_ce_loss(logits, pseudo, mask):
    """CE on target with confidence mask; unmasked pixels → ignore."""
    tgt = pseudo.clone()
    tgt[~mask] = IGNORE_INDEX
    if tgt.ne(IGNORE_INDEX).sum() == 0:
        return logits.sum() * 0.0  # zero loss but keeps computation graph
    return F.cross_entropy(logits, tgt, ignore_index=IGNORE_INDEX)


@torch.no_grad()
def make_pseudo(teacher, imgs, hw, threshold):
    logits = get_logits(teacher, imgs, hw)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf >= threshold


# ── dataset builders ─────────────────────────────────────────────────────────

def build_datasets(cfg):
    step = cfg["step"]
    crop = cfg["training"]["crop_size"]
    gta5_root = os.path.expanduser(cfg["data"]["gta5_root"])
    cs_root   = os.path.expanduser(cfg["data"]["cityscapes_root"])
    acdc_root = os.path.expanduser(cfg["data"]["acdc_root"])

    if step == 1:
        src_ds  = GTA5Dataset(gta5_root, crop_size=crop)
        tgt_ds  = CityscapesDataset(cs_root, split="train",
                                    crop_size=crop, labeled=False)
        eval_ds = CityscapesDataset(cs_root, split="val",
                                    crop_size=None, labeled=True)
    else:
        src_ds  = CityscapesDataset(cs_root, split="train",
                                    crop_size=crop, labeled=True)
        tgt_ds  = ACDCDataset(acdc_root, split="train",
                              conditions=cfg.get("acdc_conditions"),
                              crop_size=crop, labeled=False)
        eval_ds = ACDCDataset(acdc_root, split="val",
                              conditions=cfg.get("acdc_conditions"),
                              crop_size=None, labeled=True)
    return src_ds, tgt_ds, eval_ds


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, step_cfg):
    model.eval()
    step = step_cfg["step"]

    if step == 2:
        conditions = step_cfg.get("acdc_conditions", ["fog", "rain", "snow", "night"])
        cond_conf  = {c: make_confusion_matrix() for c in conditions}
        overall    = make_confusion_matrix()

        for batch in tqdm(loader, desc="Eval", leave=False):
            imgs   = batch["image"].to(device)
            labels = batch["label"].numpy()
            conds  = batch["condition"]
            B, _, H, W = imgs.shape
            logits = get_logits(model, imgs, (H, W))
            preds  = logits.argmax(1).cpu().numpy()
            for pred, label, cond in zip(preds, labels, conds):
                update_confusion_matrix(overall, pred, label)
                update_confusion_matrix(cond_conf[cond], pred, label)

        iou_all, miou_all = compute_iou(overall)
        result = {"overall": miou_all, "per_class": iou_all, "per_cond": {}}
        for cond in conditions:
            _, miou_c = compute_iou(cond_conf[cond])
            result["per_cond"][cond] = miou_c
        model.train()
        return result
    else:
        conf = make_confusion_matrix()
        for batch in tqdm(loader, desc="Eval", leave=False):
            imgs   = batch["image"].to(device)
            labels = batch["label"].numpy()
            B, _, H, W = imgs.shape
            logits = get_logits(model, imgs, (H, W))
            preds  = logits.argmax(1).cpu().numpy()
            for pred, label in zip(preds, labels):
                update_confusion_matrix(conf, pred, label)
        iou_all, miou_all = compute_iou(conf)
        model.train()
        return {"overall": miou_all, "per_class": iou_all}


def print_eval(result, cfg_step, log=print):
    iou = result["per_class"]
    log(format_iou_table(iou, result["overall"], prefix="  "))
    if cfg_step == 2 and "per_cond" in result:
        cond_str = "  Conditions: " + " | ".join(
            f"{c}={v*100:.1f}%" for c, v in result["per_cond"].items()
        )
        log(cond_str)


# ── main training loop ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--resume",  default=None,
                        help="Checkpoint .pth to load model weights from (step-2 init)")
    parser.add_argument("--smoke",   action="store_true",
                        help="50-iteration sanity check — exits without saving")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(os.path.expanduser(cfg["data"]["output_root"])) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── logging (stdout + file) ───────────────────────────────────────────────
    log_fh = open(out_dir / "train.log", "a", buffering=1)  # line-buffered

    def log(msg: str = ""):
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"Device : {device}")
    if device.type == "cuda":
        log(f"GPU    : {torch.cuda.get_device_name(0)}")
        log(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # ── datasets ──────────────────────────────────────────────────────────────
    src_ds, tgt_ds, eval_ds = build_datasets(cfg)
    tr_cfg = cfg["training"]
    bs     = 1 if args.smoke else tr_cfg["batch_size"]

    src_loader  = DataLoader(src_ds,  batch_size=bs, shuffle=True,
                             num_workers=2, drop_last=True, pin_memory=True,
                             persistent_workers=True)
    tgt_loader  = DataLoader(tgt_ds,  batch_size=bs, shuffle=True,
                             num_workers=2, drop_last=True, pin_memory=True,
                             persistent_workers=True)
    eval_loader = DataLoader(eval_ds, batch_size=1,  shuffle=False,
                             num_workers=1, pin_memory=False,
                             persistent_workers=True)

    src_iter = cycle(src_loader)
    tgt_iter = cycle(tgt_loader)

    # ── model ─────────────────────────────────────────────────────────────────
    ckpt_name = cfg["model"]["checkpoint"]
    log(f"\nLoading encoder from {ckpt_name} ...")
    student = SegformerForSemanticSegmentation.from_pretrained(
        ckpt_name,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    if args.resume:
        log(f"Initializing weights from {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        student.load_state_dict(state["model"], strict=True)

    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    student.train()

    # ── optimizer / scheduler ─────────────────────────────────────────────────
    max_iters = 50 if args.smoke else tr_cfg["max_iters"]
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=max_iters, power=1.0,
    )
    use_amp  = tr_cfg.get("mixed_precision", True)
    amp_ctx  = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp)

    threshold  = tr_cfg.get("pseudo_threshold", 0.968)
    ema_decay  = tr_cfg.get("ema_decay", 0.9999)
    lambda_u   = tr_cfg.get("lambda_u", 1.0)
    grad_accum = tr_cfg.get("grad_accum_steps", 1)
    crop       = tr_cfg["crop_size"]
    log_every  = 50 if args.smoke else tr_cfg.get("log_interval",  100)
    eval_every = max_iters if args.smoke else tr_cfg.get("eval_interval", 2000)
    save_every = max_iters if args.smoke else tr_cfg.get("save_interval", 5000)

    # ── auto-resume from mid-run checkpoint ───────────────────────────────────
    start_step  = 1
    best_miou   = 0.0
    resume_path = out_dir / "resume.pth"

    if not args.smoke and resume_path.exists():
        log(f"\nAuto-resuming from {resume_path} ...")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        student.load_state_dict(state["model"])
        teacher.load_state_dict(state["teacher"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scheduler.total_iters = max_iters  # override stale value from old checkpoints
        start_step = state["step"] + 1
        best_miou  = state["best_miou"]
        log(f"Resumed from iter {state['step']}  best_mIoU={best_miou*100:.2f}%")
        if start_step > max_iters:
            log("Training already complete — nothing to do.")
            log_fh.close()
            return

    step_name = "GTA5→CS" if cfg["step"] == 1 else "CS→ACDC"
    log(f"\nStep {cfg['step']} ({step_name}) | {max_iters} iters | "
        f"bs={bs}+{bs} | lr={tr_cfg['lr']:.1e} | amp={use_amp}")
    log(f"Source  : {len(src_ds):,} images")
    log(f"Target  : {len(tgt_ds):,} images")
    log(f"Eval on : {len(eval_ds):,} images")
    log(f"Outputs : {out_dir}")
    if start_step > 1:
        log(f"Starting at iter {start_step} (resumed)\n")
    else:
        log("")

    optimizer.zero_grad()

    for step in range(start_step, max_iters + 1):
        src_batch = next(src_iter)
        tgt_batch = next(tgt_iter)

        src_imgs   = src_batch["image"].to(device)
        src_labels = src_batch["label"].to(device)
        tgt_imgs   = tgt_batch["image"].to(device)

        with amp_ctx:
            pseudo_pred, pseudo_mask = make_pseudo(
                teacher, tgt_imgs, (crop, crop), threshold)

            s_src = get_logits(student, src_imgs, (crop, crop))
            s_tgt = get_logits(student, tgt_imgs, (crop, crop))

            loss_s = src_ce_loss(s_src, src_labels)
            loss_u = pseudo_ce_loss(s_tgt, pseudo_pred, pseudo_mask)
            loss   = (loss_s + lambda_u * loss_u) / grad_accum

        loss.backward()

        if step % grad_accum == 0:
            optimizer.step()
            scheduler.step()
            ema_update(teacher, student, ema_decay)
            optimizer.zero_grad()

        if step % log_every == 0 or step == start_step:
            ratio = pseudo_mask.float().mean().item()
            lr    = scheduler.get_last_lr()[0]
            log(f"[{step:06d}/{max_iters}]  "
                f"src={loss_s.item():.3f}  "
                f"tgt={loss_u.item():.3f}  "
                f"tot={loss_s.item() + loss_u.item():.3f}  "
                f"pseudo={ratio:.2f}  "
                f"lr={lr:.2e}")

        if step % eval_every == 0 or (step == max_iters and not args.smoke):
            log(f"\n── Eval @ iter {step} ──")
            result = evaluate(student, eval_loader, device, cfg)
            print_eval(result, cfg["step"], log=log)
            miou = result["overall"]
            log(f"  mIoU = {miou*100:.2f}%\n")

            if not args.smoke:
                if miou > best_miou:
                    best_miou = miou
                    ckpt = {"step": step, "model": student.state_dict(), "miou": miou}
                    torch.save(ckpt, out_dir / "best.pth")
                    log(f"  best.pth saved  (mIoU={miou*100:.2f}%)\n")

                # full resume checkpoint — saved after every eval
                resume_ckpt = {
                    "step":      step,
                    "model":     student.state_dict(),
                    "teacher":   teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_miou": best_miou,
                }
                tmp_path = resume_path.with_suffix(".tmp")
                torch.save(resume_ckpt, tmp_path)
                tmp_path.rename(resume_path)  # atomic on Linux; prevents corrupt resume.pth on mid-save crash
                log(f"  resume.pth saved  (iter={step})\n")

        if step % save_every == 0 and not args.smoke:
            ckpt = {"step": step, "model": student.state_dict(), "miou": best_miou}
            torch.save(ckpt, out_dir / f"ckpt_{step:06d}.pth")
            log(f"  Checkpoint saved: ckpt_{step:06d}.pth")

        # Save resume.pth every 500 iters (crash-safe: atomic rename)
        if step % 500 == 0 and step % eval_every != 0 and not args.smoke:
            resume_ckpt = {
                "step":      step,
                "model":     student.state_dict(),
                "teacher":   teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_miou": best_miou,
            }
            tmp_path = resume_path.with_suffix(".tmp")
            torch.save(resume_ckpt, tmp_path)
            tmp_path.rename(resume_path)

    if not args.smoke:
        ckpt = {"step": max_iters, "model": student.state_dict(), "miou": best_miou}
        torch.save(ckpt, out_dir / "final.pth")
        if resume_path.exists():
            resume_path.unlink()  # clean up; training is done
        log(f"\nDone.  Best mIoU={best_miou*100:.2f}%  →  {out_dir}/")
    else:
        log("\nSmoke test passed — no checkpoint saved.")

    log_fh.close()


if __name__ == "__main__":
    main()
