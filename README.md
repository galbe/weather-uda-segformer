# Weather-UDA SegFormer

MSc computer vision project — adverse-weather semantic segmentation via domain adaptation.

## Goal

Fine-tune a pretrained **SegFormer-B2** on Cityscapes (clean, labeled) and evaluate
how well it generalises to ACDC (fog / rain / snow / night). Optionally apply
pseudo-label self-training on ACDC unlabeled splits.

## Model

| Model | HF checkpoint | VRAM | Notes |
|---|---|---|---|
| **SegFormer-B2** (default) | `nvidia/segformer-b2-finetuned-cityscapes-512-1024` | ~6 GB | Best accuracy / speed trade-off on A100 |
| SegFormer-B0 (fallback) | `nvidia/segformer-b0-finetuned-cityscapes-512-1024` | ~3 GB | Use if memory is tight |

No Hugging Face login is required — both checkpoints are public.

## Dataset layout

```
~/data/
├── cityscapes/
│   ├── leftImg8bit/   train/ val/ test/
│   └── gtFine/        train/ val/ test/
├── acdc/
│   ├── rgb_anon/      fog/ rain/ snow/ night/  (train/ val/ test/)
│   └── gt/            fog/ rain/ snow/ night/  (train/ val/)
└── gta5/              (optional)
    ├── images/
    └── labels/
```

## Setup

```bash
source ~/venvs/uda/bin/activate
pip install -r requirements.txt
```

## Implementation plan

### Phase 1 — Baseline inference
Load pretrained SegFormer-B2 (already fine-tuned on Cityscapes) and run inference
on a handful of ACDC images to measure the domain gap before any adaptation.

```bash
python infer.py --config configs/paths.yaml --condition fog --num_samples 10
```

### Phase 2 — Cityscapes fine-tuning
Fine-tune the model on the full Cityscapes training set to ensure the training
pipeline is correct before touching ACDC.

```bash
python train.py --config configs/paths.yaml
```

### Phase 3 — ACDC per-weather evaluation
Evaluate mIoU on each ACDC weather condition separately to understand which
condition is hardest.

```bash
python evaluate.py --config configs/paths.yaml --split val
```

### Phase 4 — Pseudo-label adaptation (optional)
Run the model on ACDC unlabeled images, keep high-confidence predictions as
pseudo-labels, then fine-tune on the pseudo-labeled ACDC set.

```bash
python adapt.py --config configs/paths.yaml --condition fog --threshold 0.9
```

### Phase 5 — GTA5 extension (later)
Use GTA5 as additional synthetic source data. Not required for first submission.

## Project structure

```
weather-uda-segformer/
├── configs/paths.yaml      ← dataset roots and hyperparameters
├── datasets/               ← CityscapesDataset, ACDCDataset loaders
├── models/                 ← SegFormer wrapper, head utilities
├── utils/                  ← metrics, visualisation, colour maps
├── scripts/                ← slurm / shell helpers
├── outputs/                ← checkpoints, logs, prediction images
├── train.py
├── evaluate.py
├── adapt.py
└── infer.py
```

## Next command

```bash
source ~/venvs/uda/bin/activate
python infer.py --config configs/paths.yaml
```
