"""
Pseudo-label self-training on ACDC unlabeled images.
Usage:
    python adapt.py --config configs/paths.yaml --condition fog --threshold 0.9
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
    print(f"Device    : {device}")
    print(f"Condition : {args.condition}")
    print(f"Threshold : {args.threshold}")
    print("adapt.py — not yet implemented.")
    print("Complete train.py and evaluate.py first, then implement pseudo-label loop here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--condition", default="fog", choices=["fog", "rain", "snow", "night"])
    parser.add_argument("--threshold", type=float, default=0.9)
    run(parser.parse_args())
