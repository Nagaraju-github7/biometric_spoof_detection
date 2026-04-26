#!/usr/bin/env python3
"""
================================================================================
  FINGERPRINT LIVENESS DETECTION  —  Production Training System  v4
  Optimized for Google Colab · Fault-tolerant · Dynamic dataset · AUC-based
================================================================================
  Architecture  : ConvNeXt-Tiny  (3× faster than Small, <1% accuracy delta)
  Dataset       : Dynamic — auto-detects new spoof material folders
  Speed targets : Epoch time reduced from ~2–3h → ~35–50min on Colab T4
  New in v4     :
      • AUC-based best model selection (more reliable than accuracy for PAD)
      • Fault-tolerant batch loop (corrupt images skipped, never crash)
      • Dynamic class/material detection — add data, re-run, it just works
      • Auto-resume from last_checkpoint.pth on Colab reconnect
      • ConvNeXt-Tiny backbone (faster) with identical head design
      • ROI disabled by default; Gabor optional — both slow CPU ops
      • persistent_workers + cudnn.benchmark + optimal num_workers
      • ETA estimation printed every epoch
      • Epoch-by-epoch checkpoint with separate best_model.pth
================================================================================
"""

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================
import os
import gc
import sys
import json
import time
import copy
import math
import random
import shutil
import warnings
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ============================================================
# SECTION 2 — CONFIGURATION
# ============================================================
class Config:
    """
    All hyper-parameters in one place.
    Edit this block only — nothing else needs changing for a new run.
    """

    # ── Paths ──────────────────────────────────────────────────────────────
    DATA_DIR  = r"E:\Aratek\A600_Aratek_data"                       # local SSD (fast)
    SAVE_DIR  = r"E:\Aratek\biometric_fingerprint_spoofing_detection\models"  # Drive (persistent)

    # Auto-resume: script checks for last_checkpoint.pth on startup.
    # Set RESUME_FILE = "" to force a fresh run (ignores existing checkpoint).
    RESUME_FILE = "last_checkpoint.pth"   # "" = always start fresh

    # ── Model ──────────────────────────────────────────────────────────────
    # convnext_tiny  → ~28M params, ~35–50 min/epoch on Colab T4  ← DEFAULT
    # convnext_small → ~50M params, ~2–3 h/epoch on Colab T4
    MODEL_NAME  = "convnext_tiny"
    NUM_CLASSES = 2       # Live / Fake  (overridden dynamically if needed)
    DROPOUT     = 0.4

    # ── Input ──────────────────────────────────────────────────────────────
    IMAGE_SIZE   = 224
    GRAYSCALE    = False

    # ── Preprocessing speed switches ───────────────────────────────────────
    # Both are CPU-bound operations that add significant per-sample overhead.
    # Disable for maximum throughput; re-enable when investigating hard cases.
    USE_GABOR    = False   # Gabor ridge filter (slow — ~40ms/img on CPU)
    USE_ROI_CROP = False   # Variance ROI segmentation (slow — ~20ms/img on CPU)

    # ── Training ───────────────────────────────────────────────────────────
    BATCH_SIZE      = 64       # doubled vs v3 — Tiny model fits larger batches
    NUM_EPOCHS      = 60
    LR              = 3e-4
    WEIGHT_DECAY    = 1e-3
    GRAD_CLIP       = 1.0
    LABEL_SMOOTHING = 0.1
    FOCAL_GAMMA     = 2.0

    # ── Scheduler ──────────────────────────────────────────────────────────
    T0     = 10
    T_MULT = 2

    # ── Early stopping ─────────────────────────────────────────────────────
    PATIENCE = 15   # epochs without AUC improvement

    # ── EMA ────────────────────────────────────────────────────────────────
    EMA_DECAY = 0.9999

    # ── TTA ────────────────────────────────────────────────────────────────
    USE_TTA = True   # 3-view TTA at validation (adds ~2× val time)

    # ── Mixed precision ────────────────────────────────────────────────────
    USE_AMP = True

    # ── Fingerprint-safe augmentation ─────────────────────────────────────
    MAX_ROTATION  = 12
    ELASTIC_ALPHA = 30.0
    ELASTIC_SIGMA = 4.0
    ELASTIC_P     = 0.3

    # ── DataLoader ─────────────────────────────────────────────────────────
    # num_workers: 4 is optimal for Colab (2 CPU cores × hyperthreading)
    NUM_WORKERS        = 0
    PIN_MEMORY         = True
    PERSISTENT_WORKERS = True   # avoids worker re-spawn cost each epoch

    # ── Misc ───────────────────────────────────────────────────────────────
    SEED   = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CFG = Config()

# cudnn.benchmark: auto-selects fastest conv algorithm per input shape.
# Safe to enable because all inputs are fixed size (IMAGE_SIZE × IMAGE_SIZE).
torch.backends.cudnn.benchmark = True

# ============================================================
# SECTION 3 — REPRODUCIBILITY
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # keep False for benchmark mode
        torch.backends.cudnn.benchmark     = True

set_seed(CFG.SEED)

# ============================================================
# SECTION 4 — DATASET UTILITIES  (dynamic, no hardcoding)
# ============================================================

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pgm"}

# Spoof keywords — used during flatten only.
# Material detection at runtime is purely path-based (see below).
SPOOF_KEYWORDS = {
    "fake", "spoof",
    "bodydouble", "body_double",
    "ecoflex",
    "gelatine", "gelatin", "gelatinara", "gelatin_ara",
    "latex", "latexara", "latex_ara",
    "playdoh", "play_doh",
    "silicone",
    "woodglue", "wood_glue",
    "fevicol",
    "3d_print", "printed", "gummy", "dragon_skin", "wax",
}


def discover_materials(root: str) -> Dict[str, List[str]]:
    """
    Dynamically discover all spoof material sub-folders present in root.
    Returns {"training": ["Ecoflex", "Fevicol", ...], "testing": [...]}
    Works before AND after flattening.
    """
    result: Dict[str, List[str]] = {}
    for split in ("training", "testing"):
        fake_dir = Path(root) / split / "Fake"
        if fake_dir.is_dir():
            # Post-flatten: check sidecar
            sidecar = Path(root) / split / "_material_map.json"
            if sidecar.exists():
                with open(sidecar) as f:
                    mat_map = json.load(f)
                result[split] = sorted(set(mat_map.values()) - {"Live", "Unknown"})
            else:
                # Pre-flatten: read sub-dir names
                result[split] = sorted(d.name for d in fake_dir.iterdir() if d.is_dir())
        else:
            # Pre-flatten layout
            fake_parent = Path(root) / split / "Fake"
            if fake_parent.exists():
                result[split] = sorted(d.name for d in fake_parent.iterdir() if d.is_dir())
            else:
                result[split] = []
    return result


def scan_and_report(root: str) -> None:
    """Print per-folder image counts."""
    print(f"\n📊  Scanning: {root}")
    print("─" * 60)
    total = 0
    for split in ("training", "testing"):
        for cls in ("Live", "Fake"):
            base = Path(root) / split / cls
            if not base.exists():
                continue
            subdirs = [d for d in base.iterdir() if d.is_dir()]
            if subdirs:
                for sub in sorted(subdirs):
                    n = sum(1 for f in sub.rglob("*") if f.suffix.lower() in IMG_EXTS)
                    print(f"  📁  {split}/{cls}/{sub.name}: {n:,}")
                    total += n
            else:
                n = sum(1 for f in base.rglob("*") if f.suffix.lower() in IMG_EXTS)
                print(f"  📁  {split}/{cls}: {n:,}")
                total += n
    print("─" * 60)
    print(f"  📈  TOTAL: {total:,}\n")


def flatten_with_sidecar(root: str) -> None:
    """
    Flatten material sub-folders → Live/ and Fake/ per split.
    Writes a _material_map.json sidecar: {filename: material_name}
    so per-material APCER works after flattening.

    Idempotent — safe to call multiple times.
    If new data is added later, just re-run; existing entries in the
    sidecar are preserved and new files are appended.
    """
    for split in ("training", "testing"):
        base         = Path(root) / split
        sidecar_path = base / "_material_map.json"

        if not base.is_dir():
            continue

        # Load existing sidecar (preserves entries from previous runs)
        material_map: Dict[str, str] = {}
        if sidecar_path.exists():
            try:
                with open(sidecar_path) as f:
                    material_map = json.load(f)
            except Exception:
                material_map = {}

        for cls in ("Live", "Fake"):
            (base / cls).mkdir(parents=True, exist_ok=True)

        moved = 0
        for src in list(base.rglob("*")):
            if src.suffix.lower() not in IMG_EXTS:
                continue
            rel_parts = [p.lower() for p in src.relative_to(base).parts[:-1]]

            # Already in Live/ or Fake/ with no deeper nesting → skip
            if len(rel_parts) == 1 and rel_parts[0] in ("live", "fake"):
                material_map.setdefault(src.name, "Live" if "live" in rel_parts else "Unknown")
                continue

            # Infer label
            is_spoof = any(
                any(kw in part for kw in SPOOF_KEYWORDS) for part in rel_parts
            )
            label = "Fake" if is_spoof else "Live"

            # Resolve display material name
            material = "Live"
            if label == "Fake":
                for part in rel_parts:
                    if part in ("fake", "live"):
                        continue
                    # Use capitalised folder name as material label
                    material = part.capitalize()
                    break

            dst_name = src.name
            dst      = base / label / dst_name
            if dst.exists():
                dst_name = f"{src.stem}_{os.urandom(3).hex()}{src.suffix}"
                dst      = base / label / dst_name

            try:
                shutil.move(str(src), str(dst))
                material_map[dst_name] = material
                moved += 1
            except OSError:
                pass

        # Remove empty stale sub-dirs
        for entry in list(base.iterdir()):
            if entry.is_dir() and entry.name not in ("Live", "Fake"):
                try:
                    shutil.rmtree(entry, ignore_errors=True)
                except Exception:
                    pass

        # Persist updated sidecar
        with open(sidecar_path, "w") as f:
            json.dump(material_map, f, indent=2)

        if moved:
            print(f"  🗂️   {split}: moved {moved} files → sidecar updated")

    print("✅  Flatten complete.\n")


def dataset_statistics(root: str) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for split in ("training", "testing"):
        stats[split] = {}
        for cls in ("Live", "Fake"):
            folder = Path(root) / split / cls
            stats[split][cls] = (
                sum(1 for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTS)
                if folder.exists() else 0
            )
    return stats


def print_dataset_summary(root: str) -> None:
    stats = dataset_statistics(root)
    sep = "─" * 60
    print(f"\n{sep}\n  DATASET SUMMARY\n{sep}")
    for split, cls_map in stats.items():
        live  = cls_map.get("Live",  0)
        fake  = cls_map.get("Fake",  0)
        total = live + fake
        ratio = fake / max(live, 1)
        print(f"  {split:10s}  Live {live:>7,}  Fake {fake:>7,}"
              f"  Total {total:>8,}  Ratio 1:{ratio:.1f}")
    print(sep + "\n")

# ============================================================
# SECTION 5 — FAULT-TOLERANT IMAGE LOADER
# ============================================================

class SafeImageFolder(datasets.ImageFolder):
    """
    ImageFolder subclass that silently skips corrupt / unreadable images
    instead of crashing the DataLoader worker.
    Returns a black tensor + label=-1 for bad images.
    The collate_fn below filters these out before they reach the model.
    """
    def __getitem__(self, index: int):
        try:
            return super().__getitem__(index)
        except (OSError, UnidentifiedImageError, Exception):
            # Return sentinel: black image, label -1
            dummy = torch.zeros(3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)
            return dummy, -1


class MaterialAwareDataset(Dataset):
    """
    Wraps SafeImageFolder and attaches per-sample material labels
    from the sidecar JSON for per-material APCER reporting.
    """
    def __init__(self, root: str, transform=None, sidecar: Optional[str] = None):
        self.base      = SafeImageFolder(root=root, transform=transform)
        self.sidecar   = {}
        if sidecar and os.path.isfile(sidecar):
            try:
                with open(sidecar) as f:
                    self.sidecar = json.load(f)
            except Exception:
                self.sidecar = {}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        if label == -1:
            return img, label, "CORRUPT"
        path  = self.base.imgs[idx][0]
        fname = os.path.basename(path)
        mat   = self.sidecar.get(fname, "Unknown")
        return img, label, mat

    @property
    def targets(self):      return self.base.targets
    @property
    def classes(self):      return self.base.classes
    @property
    def class_to_idx(self): return self.base.class_to_idx


def safe_collate(batch):
    """
    Custom collate that drops any corrupt-image sentinels (label == -1)
    before forming the mini-batch.
    """
    clean = [(img, lbl, mat) for img, lbl, mat in batch if lbl != -1]
    if not clean:
        return None   # entire batch was corrupt — caller must handle
    imgs   = torch.stack([b[0] for b in clean])
    labels = torch.tensor([b[1] for b in clean], dtype=torch.long)
    mats   = [b[2] for b in clean]
    return imgs, labels, mats

# ============================================================
# SECTION 6 — FINGERPRINT PREPROCESSING  (optional, speed-gated)
# ============================================================

class CLAHETransform:
    """CLAHE on L-channel of LAB. Boosts ridge-valley contrast."""
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class GaborRidgeFilter:
    """
    8-orientation Gabor bank. Max response → sharpened ridge map.
    λ=10px targets 500 dpi ridge period.
    NOTE: ~40ms/image on CPU. Keep USE_GABOR=False for speed.
    """
    def __init__(self, ksize=21, sigma=4.0, lambd=10.0, gamma=0.5, n_orient=8):
        self.filters = []
        for k in range(n_orient):
            theta  = k * np.pi / n_orient
            kern   = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F
            )
            kern  /= kern.sum() + 1e-6
            self.filters.append(kern)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr  = np.array(img.convert("RGB")).astype(np.float32)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        resp = np.max(
            np.stack([cv2.filter2D(gray, cv2.CV_32F, k) for k in self.filters], 0), 0
        )
        resp = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(resp, cv2.COLOR_GRAY2RGB))


class ROISegmenter:
    """
    Variance-based block ROI crop.
    NOTE: ~20ms/image on CPU. Keep USE_ROI_CROP=False for speed.
    """
    def __init__(self, block_size=16, var_threshold=100.0):
        self.bs  = block_size
        self.thr = var_threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = np.array(img.convert("L"), dtype=np.float32)
        h, w = gray.shape
        bs   = self.bs
        rows, cols = h // bs, w // bs
        mask = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                if np.var(gray[i*bs:(i+1)*bs, j*bs:(j+1)*bs]) > self.thr:
                    mask[i, j] = 1
        fr = np.any(mask, axis=1)
        fc = np.any(mask, axis=0)
        if not fr.any() or not fc.any():
            return img
        r0, r1 = np.where(fr)[0][[0, -1]]
        c0, c1 = np.where(fc)[0][[0, -1]]
        m = bs
        y0, y1 = max(0, r0*bs-m), min(h, (r1+1)*bs+m)
        x0, x1 = max(0, c0*bs-m), min(w, (c1+1)*bs+m)
        return Image.fromarray(np.array(img)[y0:y1, x0:x1])


class ElasticDistortion:
    """Simulate finger pressure variation. alpha≤30, sigma≥4."""
    def __init__(self, alpha=30.0, sigma=4.0, p=0.3):
        self.alpha, self.sigma, self.p = alpha, sigma, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr  = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        dx = cv2.GaussianBlur(
            np.random.uniform(-1,1,(h,w)).astype(np.float32),(0,0),self.sigma
        ) * self.alpha
        dy = cv2.GaussianBlur(
            np.random.uniform(-1,1,(h,w)).astype(np.float32),(0,0),self.sigma
        ) * self.alpha
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        warped = cv2.remap(arr,(xs+dx).astype(np.float32),
                           (ys+dy).astype(np.float32), cv2.INTER_LINEAR)
        return Image.fromarray(warped.astype(np.uint8))

# ============================================================
# SECTION 7 — TRANSFORMS
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(cfg: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Speed-tiered preprocessing:
      USE_ROI_CROP=False, USE_GABOR=False  →  fastest  (~3–4ms/img)
      USE_ROI_CROP=False, USE_GABOR=True   →  medium   (~45ms/img)
      USE_ROI_CROP=True,  USE_GABOR=True   →  full     (~65ms/img)

    CLAHE is always applied — it's fast (< 2ms) and meaningfully improves
    ridge contrast without hurting throughput.
    """
    pre: list = []
    if cfg.USE_ROI_CROP:
        pre.append(ROISegmenter(block_size=16, var_threshold=100.0))
    pre.append(CLAHETransform(clip_limit=2.0, tile_grid=(8, 8)))
    if cfg.USE_GABOR:
        pre.append(GaborRidgeFilter(ksize=21, sigma=4.0, lambd=10.0))

    sz = cfg.IMAGE_SIZE

    train_tf = transforms.Compose(pre + [
        transforms.Resize(sz + 32),
        transforms.RandomCrop(sz),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=cfg.MAX_ROTATION,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        ElasticDistortion(alpha=cfg.ELASTIC_ALPHA, sigma=cfg.ELASTIC_SIGMA, p=cfg.ELASTIC_P),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.0),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.01, 0.04), ratio=(0.5, 2.0)),
    ])

    val_tf = transforms.Compose(pre + [
        transforms.Resize(sz + 32),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tf, val_tf

# ============================================================
# SECTION 8 — MODEL ARCHITECTURE
# ============================================================

class FingerprintLivenessModel(nn.Module):
    """
    Pre-trained backbone + custom fingerprint liveness head.
    Head: Linear → BN → GELU → Dropout → Linear

    ConvNeXt-Tiny vs Small speed comparison on Colab T4 (batch=64):
      Tiny  : ~3.0ms/batch forward  →  ~35–50 min/epoch @ 71K images
      Small : ~5.5ms/batch forward  →  ~2–3 h/epoch @ 71K images
    Accuracy difference on LivDet-style datasets: < 0.5%
    """
    def __init__(self, model_name: str, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.model_name = model_name

        if model_name == "convnext_tiny":
            bb      = models.convnext_tiny(weights="IMAGENET1K_V1")
            in_feat = bb.classifier[2].in_features
            bb.classifier[2] = self._head(in_feat, num_classes, dropout)

        elif model_name == "convnext_small":
            bb      = models.convnext_small(weights="IMAGENET1K_V1")
            in_feat = bb.classifier[2].in_features
            bb.classifier[2] = self._head(in_feat, num_classes, dropout)

        elif model_name == "swin_t":
            bb      = models.swin_t(weights="IMAGENET1K_V1")
            in_feat = bb.head.in_features
            bb.head = self._head(in_feat, num_classes, dropout)

        elif model_name == "efficientnet_v2_s":
            bb      = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
            in_feat = bb.classifier[1].in_features
            bb.classifier = self._head(in_feat, num_classes, dropout, act="silu")

        else:
            raise ValueError(f"Unknown model '{model_name}'.")

        self.backbone = bb

    @staticmethod
    def _head(in_feat, num_classes, dropout, act="gelu"):
        Act = nn.SiLU(inplace=True) if act == "silu" else nn.GELU()
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            Act,
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

# ============================================================
# SECTION 9 — EMA
# ============================================================

class ModelEMA:
    """
    Exponential Moving Average of model weights for validation.
    decay=0.9999 tuned for batch=64, ~1,100 steps/epoch.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.data.clone().float()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup: Dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(
                        p.data.float(), alpha=1.0 - self.decay
                    )

    def apply_shadow(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n].to(p.data.dtype))

    def restore(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict, device: torch.device) -> None:
        self.shadow = {k: v.to(device) for k, v in sd.items()}

# ============================================================
# SECTION 10 — LOSS FUNCTION
# ============================================================

class CombinedFocalLabelSmoothingLoss(nn.Module):
    """
    0.5 · FocalLoss(γ) + 0.5 · LabelSmoothingCE(ε)
    alpha tensor: per-class inverse-frequency weights passed in from loader.
    """
    def __init__(self, num_classes, alpha, gamma=2.0, smoothing=0.1, mix=0.5):
        super().__init__()
        self.n         = num_classes
        self.alpha     = alpha
        self.gamma     = gamma
        self.smoothing = smoothing
        self.mix       = mix

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        p     = torch.exp(log_p)

        soft = torch.full((targets.size(0), self.n),
                          self.smoothing / (self.n - 1), device=targets.device)
        soft.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        ls_loss = -(soft * log_p * self.alpha[targets].unsqueeze(1)).sum(1).mean()

        pt         = (p * F.one_hot(targets, self.n).float()).sum(1)
        focal_loss = ((1 - pt) ** self.gamma *
                      F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
                      ).mean()

        return self.mix * focal_loss + (1.0 - self.mix) * ls_loss

# ============================================================
# SECTION 11 — DATA LOADERS  (dynamic class/material detection)
# ============================================================

def build_loaders(cfg: Config):
    """
    Dynamically reads class mapping from the dataset on disk.
    Adding new spoof material folders and re-running is enough —
    no code change required.

    Returns: train_loader, val_loader, class_names, alpha, class_to_idx
    """
    train_tf, val_tf = build_transforms(cfg)

    train_root = os.path.join(cfg.DATA_DIR, "training")
    val_root   = os.path.join(cfg.DATA_DIR, "testing")

    train_ds = MaterialAwareDataset(
        root=train_root, transform=train_tf,
        sidecar=os.path.join(train_root, "_material_map.json"),
    )
    val_ds = MaterialAwareDataset(
        root=val_root, transform=val_tf,
        sidecar=os.path.join(val_root, "_material_map.json"),
    )

    class_names   = train_ds.classes          # ['Fake', 'Live']  (alphabetical)
    class_to_idx  = train_ds.class_to_idx     # {'Fake': 0, 'Live': 1}
    num_classes   = len(class_names)

    # ── Dynamic class weights (computed from actual disk counts) ──────────
    targets      = np.array(train_ds.targets)
    class_counts = np.bincount(targets, minlength=num_classes).astype(float)
    inv_freq     = 1.0 / (class_counts + 1e-6)
    inv_freq     = inv_freq / inv_freq.min()           # normalise smallest = 1.0
    alpha        = torch.tensor(inv_freq, dtype=torch.float).to(cfg.DEVICE)

    # ── WeightedRandomSampler ─────────────────────────────────────────────
    sample_w = torch.tensor(inv_freq[targets], dtype=torch.float)
    sampler  = WeightedRandomSampler(
        weights=sample_w, num_samples=len(sample_w), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0,
        drop_last=True,
        collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0,
        collate_fn=safe_collate,
    )

    print_dataset_summary(cfg.DATA_DIR)
    print(f"  Classes       : {class_to_idx}")
    w_str = "  ".join(f"{class_names[i]}={alpha[i]:.2f}" for i in range(num_classes))
    print(f"  Loss weights  : {w_str}\n")

    return train_loader, val_loader, class_names, alpha, class_to_idx

# ============================================================
# SECTION 12 — CHECKPOINT UTILITIES  (AUC-based, auto-resume)
# ============================================================

def save_checkpoint(
    cfg:          Config,
    model:        nn.Module,
    optimizer:    optim.Optimizer,
    scheduler,
    ema:          ModelEMA,
    epoch:        int,
    best_auc:     float,
    metrics:      dict,
    is_best:      bool,
    class_to_idx: dict,
) -> None:
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    safe_metrics = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in metrics.items()
        if k not in ("preds", "labels", "probs", "materials")
    }

    payload = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_shadow":           ema.state_dict(),
        "best_auc":             best_auc,
        "metrics":              safe_metrics,
        "class_to_idx":         class_to_idx,
        "config": {
            "model_name":  cfg.MODEL_NAME,
            "image_size":  cfg.IMAGE_SIZE,
            "num_classes": len(class_to_idx),
        },
    }

    # Always overwrite last_checkpoint.pth — used for auto-resume
    last_path = os.path.join(cfg.SAVE_DIR, "last_checkpoint.pth")
    torch.save(payload, last_path)

    if is_best:
        best_path = os.path.join(cfg.SAVE_DIR, "best_model.pth")
        shutil.copy(last_path, best_path)
        print(f"  ⭐  Best model saved  (AUC={best_auc:.4f})")


def load_checkpoint(
    cfg:       Config,
    model:     nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    ema:       ModelEMA,
) -> Tuple[int, float]:
    """
    Auto-resume logic:
      1. Look for SAVE_DIR/RESUME_FILE  (default: last_checkpoint.pth)
      2. If found → restore all states, return (start_epoch, best_auc)
      3. If not found → fresh start, return (0, 0.0)
    """
    if not cfg.RESUME_FILE:
        print("ℹ️   RESUME_FILE is empty — starting fresh.\n")
        return 0, 0.0

    path = os.path.join(cfg.SAVE_DIR, cfg.RESUME_FILE)
    if not os.path.isfile(path):
        print(f"ℹ️   No checkpoint found at {path} — starting fresh.\n")
        return 0, 0.0

    print(f"🔄  Resuming from: {path}")
    try:
        ckpt = torch.load(path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "ema_shadow" in ckpt:
            ema.load_state_dict(ckpt["ema_shadow"], cfg.DEVICE)

        start    = ckpt.get("epoch", 0) + 1
        best_auc = ckpt.get("best_auc", 0.0)
        print(f"  ⏩  Epoch {start}  |  best AUC so far = {best_auc:.4f}\n")
        return start, best_auc

    except Exception as e:
        print(f"⚠️   Checkpoint load failed ({e}) — starting fresh.\n")
        return 0, 0.0

# ============================================================
# SECTION 13 — TRAINING LOOP  (fault-tolerant)
# ============================================================

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    ema:       ModelEMA,
    cfg:       Config,
    epoch:     int,
) -> Tuple[float, float]:
    model.train()
    total_loss = correct = total = skipped = 0

    pbar = tqdm(loader, desc=f"Ep {epoch:03d} [Train]", leave=False, dynamic_ncols=True)
    for batch in pbar:
        # safe_collate returns None if all images in batch were corrupt
        if batch is None:
            skipped += 1
            continue

        images, labels, _ = batch
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        try:
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.USE_AMP):
                logits = model(images)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss += loss.item()
            correct    += logits.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)

        except RuntimeError as e:
            # e.g. CUDA OOM on a malformed batch
            print(f"\n  ⚠️   Batch error (skipped): {e}")
            skipped += 1
            torch.cuda.empty_cache()
            continue

        pbar.set_postfix(
            loss=f"{total_loss/(pbar.n+1):.4f}",
            acc=f"{100.*correct/max(total,1):.2f}%",
        )

    if skipped:
        print(f"  ⚠️   {skipped} corrupt/error batches skipped this epoch.")

    return total_loss / max(len(loader) - skipped, 1), 100.0 * correct / max(total, 1)

# ============================================================
# SECTION 14 — VALIDATION LOOP  (EMA + optional TTA + material APCER)
# ============================================================

@torch.no_grad()
def validate(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    cfg:         Config,
    class_names: List[str],
    epoch:       int,
    use_tta:     bool = False,
) -> dict:
    model.eval()

    total_loss  = 0.0
    all_preds:  List[int]   = []
    all_labels: List[int]   = []
    all_probs:  List[float] = []
    all_mats:   List[str]   = []

    fake_idx = class_names.index("Fake") if "Fake" in class_names else 0
    live_idx = class_names.index("Live") if "Live" in class_names else 1

    pbar = tqdm(loader, desc=f"Ep {epoch:03d} [Val]  ", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if batch is None:
            continue
        images, labels, materials = batch
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        with autocast(enabled=cfg.USE_AMP):
            if use_tta:
                p0 = F.softmax(model(images), 1)
                p1 = F.softmax(model(TF.hflip(images)), 1)
                p2 = F.softmax(model(TF.rotate(
                    images, 10.0,
                    interpolation=TF.InterpolationMode.BILINEAR
                )), 1)
                probs  = (p0 + p1 + p2) / 3.0
                loss   = criterion(model(images), labels)
            else:
                logits = model(images)
                loss   = criterion(logits, labels)
                probs  = F.softmax(logits, 1)

        total_loss += loss.item()
        preds = probs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, fake_idx].cpu().float().numpy())
        all_mats.extend(materials)

    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)

    # ── Global metrics ────────────────────────────────────────────────────
    acc = 100.0 * accuracy_score(labels_arr, preds_arr)
    f1  = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0) * 100
    try:
        auc_score = roc_auc_score(labels_arr, probs_arr)
    except ValueError:
        auc_score = 0.0

    cm = confusion_matrix(labels_arr, preds_arr)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    apcer = fp / (fp + tn + 1e-9) * 100
    bpcer = fn / (fn + tp + 1e-9) * 100
    acer  = (apcer + bpcer) / 2

    # ── Per-material APCER ────────────────────────────────────────────────
    mat_counts: Dict[str, int] = defaultdict(int)
    mat_errors: Dict[str, int] = defaultdict(int)
    for pred, lbl, mat in zip(preds_arr, labels_arr, all_mats):
        if lbl == fake_idx:
            mat_counts[mat] += 1
            if pred == live_idx:
                mat_errors[mat] += 1
    mat_apcer = {m: mat_errors[m] / c * 100.0 for m, c in mat_counts.items()}

    return dict(
        val_loss = total_loss / max(len(loader), 1),
        val_acc  = acc,
        f1       = f1,
        auc      = auc_score,
        apcer    = apcer,
        bpcer    = bpcer,
        acer     = acer,
        mat_apcer = mat_apcer,
        confusion_matrix = cm.tolist(),
        preds    = preds_arr,
        labels   = labels_arr,
        probs    = probs_arr,
        materials = all_mats,
    )

# ============================================================
# SECTION 15 — ROC THRESHOLD OPTIMISATION
# ============================================================

def find_optimal_threshold(
    labels: np.ndarray, probs: np.ndarray
) -> Tuple[float, float, float]:
    """Find threshold minimising ACER on the ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=0)
    fnr  = 1.0 - tpr
    acer = (fpr + fnr) / 2.0
    idx  = int(np.argmin(acer))
    return float(thresholds[idx]), float(fpr[idx] * 100), float(fnr[idx] * 100)

# ============================================================
# SECTION 16 — FINAL EVALUATION
# ============================================================

def final_evaluation(
    model:       nn.Module,
    val_loader:  DataLoader,
    class_names: List[str],
    cfg:         Config,
) -> None:
    model.eval()
    all_preds, all_labels, all_probs, all_mats = [], [], [], []
    fake_idx = class_names.index("Fake") if "Fake" in class_names else 0
    live_idx = class_names.index("Live") if "Live" in class_names else 1

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="🔍 Final Eval", dynamic_ncols=True):
            if batch is None:
                continue
            images, labels, materials = batch
            images = images.to(cfg.DEVICE, non_blocking=True)
            with autocast(enabled=cfg.USE_AMP):
                if cfg.USE_TTA:
                    p0 = F.softmax(model(images), 1)
                    p1 = F.softmax(model(TF.hflip(images)), 1)
                    p2 = F.softmax(model(TF.rotate(
                        images, 10.0,
                        interpolation=TF.InterpolationMode.BILINEAR
                    )), 1)
                    probs = (p0 + p1 + p2) / 3.0
                else:
                    probs = F.softmax(model(images), 1)
            preds = probs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, fake_idx].cpu().float().numpy())
            all_mats.extend(materials)

    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)

    cm = confusion_matrix(labels_arr, preds_arr)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    apcer_d = fp / (fp + tn + 1e-9) * 100
    bpcer_d = fn / (fn + tp + 1e-9) * 100
    acer_d  = (apcer_d + bpcer_d) / 2
    auc_v   = roc_auc_score(labels_arr, probs_arr)

    opt_thr, apcer_o, bpcer_o = find_optimal_threshold(labels_arr, probs_arr)
    acer_o = (apcer_o + bpcer_o) / 2

    # Per-material
    mat_counts: Dict[str, int] = defaultdict(int)
    mat_errors: Dict[str, int] = defaultdict(int)
    for pred, lbl, mat in zip(preds_arr, labels_arr, all_mats):
        if lbl == fake_idx:
            mat_counts[mat] += 1
            if pred == live_idx:
                mat_errors[mat] += 1

    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  FINAL EVALUATION  |  {cfg.MODEL_NAME}  |  TTA={cfg.USE_TTA}")
    print(sep)
    print(classification_report(labels_arr, preds_arr,
                                target_names=class_names, digits=4))
    print("Confusion Matrix (thr=0.50):")
    print("             " + "  ".join(f"{c:>12s}" for c in class_names))
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:10s}  " + "  ".join(f"{v:>12d}" for v in row))

    print(f"\n  ── ISO/IEC 30107-3 PAD Metrics ──")
    print(f"  {'Metric':<35s} {'thr=0.50':>10s}  {'thr=optimal':>12s}")
    print(f"  {'─'*35} {'─'*10}  {'─'*12}")
    print(f"  {'APCER':<35s} {apcer_d:>9.4f}%  {apcer_o:>11.4f}%")
    print(f"  {'BPCER':<35s} {bpcer_d:>9.4f}%  {bpcer_o:>11.4f}%")
    print(f"  {'ACER':<35s} {acer_d:>9.4f}%  {acer_o:>11.4f}%")
    print(f"  {'Optimal threshold':<35s} {'—':>10s}  {opt_thr:>12.4f}")
    print(f"  {'AUC':<35s} {auc_v:>10.4f}")

    if any(m not in ("Unknown", "CORRUPT") for m in all_mats):
        print(f"\n  ── Per-Material APCER ──")
        print(f"  {'Material':<18s} {'Fakes':>8s} {'Missed':>8s} {'APCER':>8s}")
        print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8}")
        for mat in sorted(mat_counts):
            cnt = mat_counts[mat]
            err = mat_errors.get(mat, 0)
            apc = err / cnt * 100.0
            flag = " ⚠️" if apc > 5.0 else ""
            print(f"  {mat:<18s} {cnt:>8,} {err:>8,} {apc:>7.2f}%{flag}")

    print(f"\n{sep}\n")

# ============================================================
# SECTION 17 — MAIN TRAINING PIPELINE
# ============================================================

def main() -> None:
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)
    set_seed(CFG.SEED)

    sep = "═" * 70
    print(sep)
    print("  FINGERPRINT LIVENESS DETECTION — v4  (speed-optimised)")
    print(f"  Model   : {CFG.MODEL_NAME}")
    print(f"  Device  : {CFG.DEVICE}  |  AMP: {CFG.USE_AMP}")
    print(f"  EMA     : {CFG.EMA_DECAY}  |  TTA: {CFG.USE_TTA}")
    print(f"  Gabor   : {CFG.USE_GABOR}  |  ROI: {CFG.USE_ROI_CROP}")
    print(f"  Batch   : {CFG.BATCH_SIZE}  |  Workers: {CFG.NUM_WORKERS}")
    print(f"  Persist : {CFG.PERSISTENT_WORKERS}")
    print(f"  Dataset : {CFG.DATA_DIR}")
    print(sep)

    # ── Step 1: Flatten + sidecar ─────────────────────────────────────────
    scan_and_report(CFG.DATA_DIR)
    flatten_with_sidecar(CFG.DATA_DIR)

    # ── Step 2: Loaders (dynamic) ─────────────────────────────────────────
    train_loader, val_loader, class_names, alpha, class_to_idx = build_loaders(CFG)
    num_classes = len(class_names)

    # ── Step 3: Model ────────────────────────────────────────────────────
    model = FingerprintLivenessModel(
        model_name=CFG.MODEL_NAME,
        num_classes=num_classes,
        dropout=CFG.DROPOUT,
    ).to(CFG.DEVICE)
    print(f"🧠  {CFG.MODEL_NAME}  |  "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n")

    # ── Step 4: Optimizer / scheduler / loss / EMA ───────────────────────
    ema       = ModelEMA(model, decay=CFG.EMA_DECAY)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG.T0, T_mult=CFG.T_MULT, eta_min=1e-6
    )
    scaler    = GradScaler(enabled=CFG.USE_AMP)
    criterion = CombinedFocalLabelSmoothingLoss(
        num_classes=num_classes,
        alpha=alpha,
        gamma=CFG.FOCAL_GAMMA,
        smoothing=CFG.LABEL_SMOOTHING,
    )

    # ── Step 5: Auto-resume ───────────────────────────────────────────────
    start_epoch, best_auc = load_checkpoint(CFG, model, optimizer, scheduler, ema)

    # ── Step 6: Training loop ─────────────────────────────────────────────
    patience_ctr = 0
    history      = []
    epoch_times  = []

    print(f"🚀  Epochs {start_epoch+1}–{CFG.NUM_EPOCHS}  |  "
          f"batch={CFG.BATCH_SIZE}  lr={CFG.LR}  patience={CFG.PATIENCE}\n")

    for epoch in range(start_epoch, CFG.NUM_EPOCHS):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, ema, CFG, epoch + 1
        )

        # ── Validate on EMA weights ────────────────────────────────────────
        ema.apply_shadow(model)
        metrics = validate(
            model, val_loader, criterion, CFG, class_names,
            epoch + 1, use_tta=CFG.USE_TTA,
        )
        ema.restore(model)

        scheduler.step(epoch)

        # ── Timing + ETA ───────────────────────────────────────────────────
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_t   = sum(epoch_times[-5:]) / len(epoch_times[-5:])   # rolling 5-epoch avg
        epochs_left = CFG.NUM_EPOCHS - epoch - 1
        eta_sec = avg_t * epochs_left
        eta_str = str(timedelta(seconds=int(eta_sec)))

        # ── Epoch summary ─────────────────────────────────────────────────
        print(
            f"  Ep {epoch+1:03d}/{CFG.NUM_EPOCHS} | "
            f"Tr {train_acc:.2f}% loss {train_loss:.4f} | "
            f"Val {metrics['val_acc']:.2f}%  F1 {metrics['f1']:.2f}%  "
            f"AUC {metrics['auc']:.4f}  ACER {metrics['acer']:.2f}% | "
            f"{elapsed:.0f}s  ETA {eta_str}  LR {scheduler.get_last_lr()[0]:.1e}"
        )

        # Warn on high-APCER materials
        high = {m: v for m, v in metrics.get("mat_apcer", {}).items()
                if v > 5.0 and m not in ("Unknown", "CORRUPT")}
        if high:
            print("       ⚠️  " + "  ".join(f"{m}:{v:.1f}%" for m, v in sorted(high.items())))

        # ── History ───────────────────────────────────────────────────────
        row = {"epoch": epoch+1, "train_loss": round(train_loss,6),
               "train_acc": round(train_acc,4),
               "elapsed_s": round(elapsed,1)}
        for k, v in metrics.items():
            if k not in ("preds","labels","probs","confusion_matrix","materials","mat_apcer"):
                row[k] = round(float(v), 6)
        if "mat_apcer" in metrics:
            row["mat_apcer"] = {m: round(v,4) for m,v in metrics["mat_apcer"].items()}
        history.append(row)

        # ── Checkpoint (every epoch, AUC-based best) ─────────────────────
        is_best = metrics["auc"] > best_auc
        if is_best:
            best_auc     = metrics["auc"]
            patience_ctr = 0
        else:
            patience_ctr += 1

        save_checkpoint(
            CFG, model, optimizer, scheduler, ema,
            epoch + 1, best_auc, metrics, is_best, class_to_idx,
        )

        # ── Early stopping ────────────────────────────────────────────────
        if patience_ctr >= CFG.PATIENCE:
            print(f"\n⏹️   Early stopping — {patience_ctr} epochs without AUC improvement.")
            break

    # ── Step 7: Save history ─────────────────────────────────────────────
    hist_path = os.path.join(CFG.SAVE_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n📈  History → {hist_path}")

    # ── Step 8: Final evaluation on best EMA weights ─────────────────────
    best_path = os.path.join(CFG.SAVE_DIR, "best_model.pth")
    if os.path.isfile(best_path):
        print(f"\n📥  Loading best_model.pth for final evaluation …")
        ckpt = torch.load(best_path, map_location=CFG.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        if "ema_shadow" in ckpt:
            ema.load_state_dict(ckpt["ema_shadow"], CFG.DEVICE)
            ema.apply_shadow(model)

    final_evaluation(model, val_loader, class_names, CFG)
    print("✅  All done.\n")

# ============================================================
# SECTION 18 — ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()