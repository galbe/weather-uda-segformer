"""
Fine-tune SegFormer-B2 on Cityscapes.
Usage:
    python train.py --config configs/paths.yaml
"""
import argparse
import os
from pathlib import Path

import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run(args):
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print("train.py — not yet implemented.")
    print("Next step: implement CityscapesDataset in datasets/ then wire it here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    run(parser.parse_args())
