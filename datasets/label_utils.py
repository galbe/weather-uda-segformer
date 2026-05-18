"""
Label ID lookup tables for GTA5, Cityscapes, and ACDC.

GTA5    → class IDs 0–33   (stored as palette index in P-mode PNG)
CS/ACDC → city IDs 0–33    (stored as grayscale L-mode PNG)
Both need to be mapped to Cityscapes trainIds 0–18 (255 = ignore).
"""
import numpy as np

IGNORE_INDEX = 255
NUM_CLASSES  = 19

# GTA5 class index (0-33) → Cityscapes trainId
_GTA5_MAP = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7:  0,   # road
    8:  1,   # sidewalk
    9: 255, 10: 255,
    11:  2,  # building
    12:  3,  # wall
    13:  4,  # fence
    14: 255, 15: 255, 16: 255,
    17:  5,  # pole
    18: 255,
    19:  6,  # traffic light
    20:  7,  # traffic sign
    21:  8,  # vegetation
    22:  9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255, 30: 255,
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}

# Cityscapes / ACDC city ID → trainId
_CS_MAP = {
    -1: 255, 255: 255,
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7:  0,   # road
    8:  1,   # sidewalk
    9: 255, 10: 255,
    11:  2,  # building
    12:  3,  # wall
    13:  4,  # fence
    14: 255, 15: 255, 16: 255,
    17:  5,  # pole
    18: 255,
    19:  6,  # traffic light
    20:  7,  # traffic sign
    21:  8,  # vegetation
    22:  9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255, 30: 255,
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}

TRAINID_TO_NAME = {
     0: "road",        1: "sidewalk",   2: "building",
     3: "wall",        4: "fence",      5: "pole",
     6: "traffic light",               7: "traffic sign",
     8: "vegetation",  9: "terrain",   10: "sky",
    11: "person",     12: "rider",     13: "car",
    14: "truck",      15: "bus",       16: "train",
    17: "motorcycle", 18: "bicycle",
}


def build_gta5_lut() -> np.ndarray:
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for k, v in _GTA5_MAP.items():
        if 0 <= k < 256:
            lut[k] = v
    return lut


def build_cs_lut() -> np.ndarray:
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for k, v in _CS_MAP.items():
        if 0 <= k < 256:
            lut[k] = v
    return lut
