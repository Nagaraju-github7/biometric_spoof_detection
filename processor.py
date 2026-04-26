# processor.py
"""
Fingerprint image preprocessing and inference pipeline for the Aratek A600 scanner.

Hardware specs
--------------
Sensor resolution : 1600 x 1200 pixels
Output            : 500 dpi grayscale fingerprint image

Pipeline
--------
1.  Accept input as PIL.Image, numpy array, or file path.
2.  Convert to grayscale.
3.  Apply CLAHE (clipLimit=2.0, tileGridSize=(8,8)) for ridge enhancement.
4.  DO NOT resize – work at native 1600×1200 resolution.
5.  Extract a 224×224 centre crop aligned to the sensor's optical centre.
6.  Replicate the single grayscale channel to 3 channels (RGB-like tensor).
7.  Normalise with ImageNet statistics so weights transfer cleanly from
    ImageNet-pretrained ResNet18 backbones.
8.  Run inference and return a structured prediction result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Aratek A600 native sensor resolution
A600_WIDTH: int = 1600
A600_HEIGHT: int = 1200

# Centre-crop target (must match classifier input size)
CROP_SIZE: int = 224
HALF_CROP: int = CROP_SIZE // 2  # 112

# ImageNet normalisation (matches ResNet18 pre-training)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# CLAHE parameters tuned for 500 dpi fingerprint ridges
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)

# Class index → human-readable label (matches training convention)
CLASS_LABELS: dict[int, str] = {0: "Spoof", 1: "Live"}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Structured output from :func:`predict_spoof_live`."""

    label: str          # "Live" or "Spoof"
    class_index: int    # 1 = Live, 0 = Spoof
    confidence: float   # probability of the winning class  [0, 1]
    prob_live: float    # P(Live)  – class index 1
    prob_spoof: float   # P(Spoof) – class index 0
    inference_ms: float # wall-clock inference time in milliseconds

    def __str__(self) -> str:
        return (
            f"PredictionResult("
            f"label={self.label!r}, "
            f"confidence={self.confidence:.4f}, "
            f"prob_live={self.prob_live:.4f}, "
            f"prob_spoof={self.prob_spoof:.4f}, "
            f"inference_ms={self.inference_ms:.1f})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_gray(image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Accept any reasonable input type and return a **uint8 grayscale**
    numpy array without changing spatial dimensions.

    Parameters
    ----------
    image:
        File path, PIL Image, or numpy array (BGR or RGB, or already gray).

    Returns
    -------
    np.ndarray
        Shape ``(H, W)``, dtype ``uint8``.
    """
    if isinstance(image, (str, Path)):
        arr = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"Could not read image from path: {image}")
        return arr

    if isinstance(image, Image.Image):
        return np.array(image.convert("L"))

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            # Already grayscale
            return image.astype(np.uint8)
        if image.ndim == 3:
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            if image.shape[2] == 1:
                return image[:, :, 0].astype(np.uint8)

    raise TypeError(
        f"Unsupported image type: {type(image)}. "
        "Expected str, Path, PIL.Image, or numpy.ndarray."
    )


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalisation to enhance
    fingerprint ridge contrast at 500 dpi.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image, shape ``(H, W)``, dtype ``uint8``.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced grayscale image, same shape and dtype.
    """
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    return clahe.apply(gray)


def _centre_crop_224(gray: np.ndarray) -> np.ndarray:
    """
    Extract a 224×224 centre crop from a grayscale image **without
    resizing**.  Computed from the image's own centre, which for the
    Aratek A600 (1600×1200) is pixel (800, 600) – the optical centre of
    the platen.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Shape ``(224, 224)``, dtype ``uint8``.

    Raises
    ------
    ValueError
        If the image is too small to yield a 224×224 crop.
    """
    height, width = gray.shape[:2]

    if height < CROP_SIZE or width < CROP_SIZE:
        raise ValueError(
            f"Image ({width}×{height}) is too small for a {CROP_SIZE}×{CROP_SIZE} "
            "centre crop.  The Aratek A600 produces 1600×1200 images; verify the "
            "input source."
        )

    center_x: int = width // 2
    center_y: int = height // 2

    crop_x1: int = center_x - HALF_CROP   # 800 - 112 = 688  (A600 default)
    crop_y1: int = center_y - HALF_CROP   # 600 - 112 = 488
    crop_x2: int = center_x + HALF_CROP   # 800 + 112 = 912
    crop_y2: int = center_y + HALF_CROP   # 600 + 112 = 712

    return gray[crop_y1:crop_y2, crop_x1:crop_x2]


# Reusable normalisation transform (constructed once at import time)
_NORMALISE: transforms.Compose = transforms.Compose([
    transforms.ToTensor(),                          # (H, W, C) uint8 → (C, H, W) float [0,1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_a600_image(
    image: Union[str, Path, np.ndarray, Image.Image],
) -> torch.Tensor:
    """
    Full Aratek A600 preprocessing pipeline.

    Steps
    -----
    1. Convert input to grayscale ``(H, W)`` numpy array.
    2. Apply CLAHE for ridge contrast enhancement.
    3. Extract 224×224 centre crop (no resize).
    4. Stack the single channel three times → ``(224, 224, 3)`` uint8.
    5. Apply ImageNet normalisation → ``(3, 224, 224)`` float32 tensor.

    Parameters
    ----------
    image:
        Raw fingerprint image from the A600 sensor.  Accepts a file path,
        PIL Image, or numpy array (grayscale, BGR, or RGBA).

    Returns
    -------
    torch.Tensor
        Shape ``(3, 224, 224)``, dtype ``float32``, ready for batching.

    Raises
    ------
    TypeError  : Unsupported input type.
    ValueError : Image too small for centre crop.
    RuntimeError : Any unexpected processing failure.
    """
    try:
        # Step 1 – grayscale conversion
        gray = _to_numpy_gray(image)

        # Step 2 – CLAHE ridge enhancement
        gray = _apply_clahe(gray)

        # Step 3 – 224×224 centre crop (native resolution, no resize)
        crop = _centre_crop_224(gray)

        # Step 4 – replicate to 3-channel so ImageNet weights apply cleanly
        rgb_crop = np.stack([crop, crop, crop], axis=-1)  # (224, 224, 3) uint8

        # Step 5 – normalise to float tensor
        tensor = _NORMALISE(rgb_crop)  # (3, 224, 224) float32

        return tensor

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error during A600 preprocessing: {exc}"
        ) from exc


def run_inference(
    model: nn.Module,
    device: torch.device,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward pass through *model* on a pre-processed tensor.

    Parameters
    ----------
    model:
        A loaded, eval-mode ``nn.Module`` (e.g. from ``models.get_model_by_name``).
    device:
        Device the model lives on.
    tensor:
        Pre-processed image tensor, shape ``(3, 224, 224)``.

    Returns
    -------
    probabilities : torch.Tensor
        Shape ``(2,)`` – softmax class probabilities on CPU.
    logits : torch.Tensor
        Shape ``(2,)`` – raw model output on CPU (useful for calibration).

    Raises
    ------
    RuntimeError : Forward-pass failure.
    """
    try:
        # Add batch dimension and move to model device
        batch: torch.Tensor = tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)

        with torch.no_grad():
            logits: torch.Tensor = model(batch)              # (1, 2)
            probabilities: torch.Tensor = torch.softmax(logits, dim=1)  # (1, 2)

        return probabilities[0].cpu(), logits[0].cpu()

    except Exception as exc:
        raise RuntimeError(f"Inference forward pass failed: {exc}") from exc


def predict_spoof_live(
    model: nn.Module,
    device: torch.device,
    image: Union[str, Path, np.ndarray, Image.Image],
) -> PredictionResult:
    """
    End-to-end spoof / live prediction for a single Aratek A600 fingerprint.

    Internally calls :func:`preprocess_a600_image` then :func:`run_inference`
    and packages the outputs into a :class:`PredictionResult`.

    Parameters
    ----------
    model:
        Loaded, eval-mode ``nn.Module``.
    device:
        Device the model lives on.
    image:
        Raw input – file path, PIL Image, or numpy array.

    Returns
    -------
    PredictionResult
        Dataclass with label, class index, per-class probabilities, and
        wall-clock inference time.

    Raises
    ------
    TypeError    : Unsupported image input.
    ValueError   : Image too small for the A600 centre crop.
    RuntimeError : Preprocessing or inference failure.

    Example
    -------
    >>> from models import get_model_by_name
    >>> from processor import predict_spoof_live
    >>> model, device = get_model_by_name("models/Nagaraju_Final_ResNet.pth")
    >>> result = predict_spoof_live(model, device, "scan.png")
    >>> print(result)
    PredictionResult(label='Live', confidence=0.9821, ...)
    """
    t_start: float = time.perf_counter()

    # --- preprocessing ---
    tensor: torch.Tensor = preprocess_a600_image(image)

    # --- inference ---
    probs, _ = run_inference(model, device, tensor)

    t_elapsed_ms: float = (time.perf_counter() - t_start) * 1_000.0

    # --- parse results ---
    prob_spoof: float = float(probs[0].item())
    prob_live: float = float(probs[1].item())

    class_index: int = int(torch.argmax(probs).item())
    confidence: float = float(probs[class_index].item())
    label: str = CLASS_LABELS[class_index]

    return PredictionResult(
        label=label,
        class_index=class_index,
        confidence=confidence,
        prob_live=prob_live,
        prob_spoof=prob_spoof,
        inference_ms=t_elapsed_ms,
    )