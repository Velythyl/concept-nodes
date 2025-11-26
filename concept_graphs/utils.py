import json
import os
import torch
import numpy as np
import random
import logging
from .mapping.ObjectMap import ObjectMap
import pickle
import re
import open3d as o3d
from pathlib import Path

# A logger for this file
log = logging.getLogger(__name__)


def load_point_cloud(path):
    path = Path(path)
    pcd = o3d.io.read_point_cloud(str(path / "point_cloud.pcd"))

    with open(path / "segments_anno.json", "r") as f:
        segments_anno = json.load(f)

    # Build a pcd with random colors
    pcd_o3d = []

    for ann in segments_anno["segGroups"]:
        obj = pcd.select_by_index(ann["segments"])
        pcd_o3d.append(obj)

    return pcd_o3d


def set_seed(seed: int = 42) -> None:
    # From wanb https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info(f"Random seed set as {seed}")


def load_map(path: str) -> ObjectMap:
    map = pickle.load(open(path, "rb"))

    for obj in map:
        obj.pcd_to_o3d()

    return map


def split_camel_preserve_acronyms(name):
    # Insert space between lowercase → uppercase
    # OR between acronym → normal word
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return s.lower()
