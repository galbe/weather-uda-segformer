"""
Quick inference: load pretrained SegFormer-B2 and run on a few ACDC images.
Usage:
    python infer.py --config configs/paths.yaml --condition fog --num_samples 5
"""
import argparse
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F


CHECKPOINT = "nvidia/segformer-b2-finetuned-cityscapes-512-1024"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def collect_images(acdc_root, condition, split="val", n=5):
    base = Path(acdc_root) / "rgb_anon" / condition / split
    paths = sorted(base.rglob("*.png"))[:n]
    if not paths:
        raise FileNotFoundError(f"No images found under {base}")
    return paths


def run(args):
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    print(f"Loading {CHECKPOINT} ...")
    processor = SegformerImageProcessor.from_pretrained(CHECKPOINT)
    model = SegformerForSemanticSegmentation.from_pretrained(CHECKPOINT).to(device)
    model.eval()

    acdc_root = os.path.expanduser(cfg["data"]["acdc_root"])
    images = collect_images(acdc_root, args.condition, n=args.num_samples)
    print(f"Running on {len(images)} images from ACDC/{args.condition} ...")

    out_dir = Path(os.path.expanduser(cfg["data"]["output_root"])) / "infer" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits          # (1, C, H/4, W/4)
        upsampled = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        pred = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        save_path = out_dir / img_path.name
        Image.fromarray(pred.astype("uint8")).save(save_path)
        print(f"  saved {save_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--condition", default="fog", choices=["fog", "rain", "snow", "night"])
    parser.add_argument("--num_samples", type=int, default=5)
    run(parser.parse_args())
