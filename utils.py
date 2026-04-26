# utils.py
"""
Shared utility functions for the Fingerprint Spoof Detection system.

Responsibilities
----------------
- Dataset introspection  : flatten_dataset()
- User feedback storage  : save_user_feedback()
- Structured event log   : log_event()

Feedback directory layout
-------------------------
./user_feedback/
    Live/
        <stem>_<timestamp>.png
        <stem>_<timestamp>.json
    Spoof/
        <material>/               # optional sub-folder when material is known
            <stem>_<timestamp>.png
            <stem>_<timestamp>.json
    events.log                    # append-only NDJSON event log
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level logger (plain text; callers may attach their own handlers)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEEDBACK_ROOT: Path = Path("./user_feedback")
EVENT_LOG_FILE: Path = FEEDBACK_ROOT / "events.log"

# Supported raster image extensions (lower-case)
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
)

# Feedback source tag embedded in every metadata JSON
FEEDBACK_SOURCE: str = "streamlit_app"

# Class index → folder name  (must match training convention: 0=Spoof, 1=Live)
LABEL_FOLDER: dict[int, str] = {0: "Spoof", 1: "Live"}
LABEL_NAME: dict[int, str] = {0: "Spoof", 1: "Live"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with 'Z' suffix."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _timestamp_slug() -> str:
    """Return a filesystem-safe timestamp slug, e.g. ``20240315_143022_456``."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _ensure_dir(path: Path) -> Path:
    """Create *path* (and any parents) if it does not exist; return *path*."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_pil_image(image: Union[Image.Image, np.ndarray], dest: Path) -> None:
    """
    Save *image* to *dest* as a lossless PNG regardless of input type.

    Parameters
    ----------
    image:
        PIL Image or numpy array (uint8, shape HxW or HxWxC).
    dest:
        Target file path; the ``.png`` extension is enforced by the caller.
    """
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        raise TypeError(
            f"Cannot save image of type {type(image).__name__}. "
            "Expected PIL.Image or numpy.ndarray."
        )

    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")

    pil.save(str(dest), format="PNG", optimize=False)


def _append_event_log(payload: dict[str, Any]) -> None:
    """
    Append *payload* as a single NDJSON line to :data:`EVENT_LOG_FILE`.
    Failures are swallowed so that a logging error never breaks a user flow.
    """
    try:
        _ensure_dir(FEEDBACK_ROOT)
        with EVENT_LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:  # pragma: no cover
        logger.warning("Could not append to event log:\n%s", traceback.format_exc())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flatten_dataset(
    root: Union[str, Path],
    *,
    extensions: frozenset[str] = IMAGE_EXTENSIONS,
) -> list[dict[str, Any]]:
    """
    Recursively walk *root* and return a flat list of image records.

    Each record is a dictionary with the following keys:

    +-----------------+-------------------------------------------------------+
    | Key             | Value                                                 |
    +=================+=======================================================+
    | ``path``        | Absolute ``Path`` to the image file                   |
    +-----------------+-------------------------------------------------------+
    | ``relative``    | Path relative to *root*                               |
    +-----------------+-------------------------------------------------------+
    | ``material``    | Name of the immediate parent directory                |
    +-----------------+-------------------------------------------------------+
    | ``label_folder``| Top-level subfolder name (e.g. ``"Live"`` / ``"Fake"``)|
    +-----------------+-------------------------------------------------------+
    | ``true_label``  | ``1`` if *label_folder* is ``"live"``, else ``0``     |
    +-----------------+-------------------------------------------------------+
    | ``filename``    | Bare filename (stem + suffix)                         |
    +-----------------+-------------------------------------------------------+

    Parameters
    ----------
    root:
        Dataset root directory.  Expected layouts::

            root/
                Live/
                    img001.png
                Fake/
                    Ecoflex/
                        img001.png
                    Gelatin/
                        img002.png

        or a flat layout::

            root/
                img001.png
                img002.png

    extensions:
        Set of lower-case extensions to include (default: common raster
        formats).  Pass ``frozenset({".png"})`` to restrict to PNG only.

    Returns
    -------
    list[dict[str, Any]]
        One record per image file discovered under *root*.

    Raises
    ------
    FileNotFoundError
        If *root* does not exist.
    """
    root = Path(root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    records: list[dict[str, Any]] = []

    def _iter_images(directory: Path) -> Iterator[Path]:
        for item in directory.rglob("*"):
            if item.is_file() and item.suffix.lower() in extensions:
                yield item

    for img_path in sorted(_iter_images(root)):
        relative = img_path.relative_to(root)
        parts = relative.parts  # e.g. ("Live", "img001.png") or ("Fake", "Ecoflex", "img.png")

        label_folder: str = parts[0] if len(parts) > 1 else ""
        material: str = (
            parts[-2]           # immediate parent, e.g. "Ecoflex" or "Live"
            if len(parts) >= 2
            else root.name
        )
        true_label: int = 1 if label_folder.lower() == "live" else 0

        records.append(
            {
                "path": img_path,
                "relative": relative,
                "material": material,
                "label_folder": label_folder,
                "true_label": true_label,
                "filename": img_path.name,
            }
        )

    logger.debug(
        "flatten_dataset: found %d image(s) under '%s'", len(records), root
    )
    return records


def save_user_feedback(
    image: Union[Image.Image, np.ndarray],
    original_filename: Union[str, Path],
    predicted_label: int,
    correct_label: int,
    confidence: float,
    material_type: str | None = None,
    *,
    feedback_root: Union[str, Path] = FEEDBACK_ROOT,
    source: str = FEEDBACK_SOURCE,
) -> tuple[bool, str]:
    """
    Persist a misclassified (or user-corrected) fingerprint image together
    with its metadata JSON sidecar.

    Directory layout::

        <feedback_root>/
            Live/
                <stem>_<timestamp>.png
                <stem>_<timestamp>.json
            Spoof/
                <material>/           ← only created when material_type is given
                    <stem>_<timestamp>.png
                    <stem>_<timestamp>.json

    Metadata JSON schema::

        {
            "timestamp":       "2024-03-15T14:30:22.456Z",
            "predicted_label": "Spoof",
            "correct_label":   "Live",
            "filename":        "scan_20240315_143022_456.png",
            "source":          "streamlit_app",
            "confidence":      0.8731,
            "material_type":   null,
            "original_filename": "upload_001.png"
        }

    Parameters
    ----------
    image:
        The fingerprint image to save (PIL Image or numpy array).
    original_filename:
        Original name / path of the uploaded or captured file (used as the
        stem for the saved file and recorded in metadata).
    predicted_label:
        Model output class index (0 = Spoof, 1 = Live).
    correct_label:
        Ground-truth class index supplied by the user (0 = Spoof, 1 = Live).
    confidence:
        Model confidence for *predicted_label* in ``[0, 1]``.
    material_type:
        Optional spoof material name (e.g. ``"Ecoflex"``).  When provided
        and *correct_label* is Spoof (0), a sub-folder is created.
    feedback_root:
        Root directory for feedback storage (default: ``./user_feedback``).
    source:
        Tag identifying the calling application, embedded in the JSON.

    Returns
    -------
    success : bool
        ``True`` on success, ``False`` on any failure.
    detail : str
        Absolute path to the saved PNG on success; error message on failure.
    """
    try:
        feedback_root = Path(feedback_root)
        stem = Path(original_filename).stem
        ts = _timestamp_slug()
        save_name = f"{stem}_{ts}"

        # Determine target directory
        label_name: str = LABEL_NAME.get(correct_label, "Unknown")

        if correct_label == 0 and material_type:
            # Spoof with known material → sub-folder
            target_dir = _ensure_dir(feedback_root / "Spoof" / material_type)
        else:
            target_dir = _ensure_dir(feedback_root / label_name)

        # ----------------------------------------------------------------
        # Save image
        # ----------------------------------------------------------------
        image_path = target_dir / f"{save_name}.png"
        _save_pil_image(image, image_path)

        # ----------------------------------------------------------------
        # Save metadata JSON sidecar
        # ----------------------------------------------------------------
        metadata: dict[str, Any] = {
            "timestamp": _utc_now(),
            "predicted_label": LABEL_NAME.get(predicted_label, str(predicted_label)),
            "correct_label": label_name,
            "filename": image_path.name,
            "source": source,
            "confidence": round(float(confidence), 6),
            "material_type": material_type,
            "original_filename": str(original_filename),
        }

        json_path = target_dir / f"{save_name}.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)

        # ----------------------------------------------------------------
        # Structured event log entry
        # ----------------------------------------------------------------
        log_event(
            event_type="user_feedback_saved",
            details={
                "image_path": str(image_path),
                "json_path": str(json_path),
                "predicted_label": metadata["predicted_label"],
                "correct_label": metadata["correct_label"],
                "confidence": metadata["confidence"],
                "material_type": material_type,
                "source": source,
            },
        )

        logger.info(
            "Feedback saved → %s  (predicted=%s  correct=%s)",
            image_path.name,
            metadata["predicted_label"],
            metadata["correct_label"],
        )
        return True, str(image_path)

    except Exception as exc:  # pragma: no cover
        error_msg = f"save_user_feedback failed: {exc}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return False, error_msg


def log_event(
    event_type: str,
    details: dict[str, Any] | None = None,
    *,
    level: str = "INFO",
    feedback_root: Union[str, Path] = FEEDBACK_ROOT,
) -> None:
    """
    Append a structured NDJSON event record to ``<feedback_root>/events.log``.

    Each line written to the log file is a self-contained JSON object::

        {
            "timestamp": "2024-03-15T14:30:22.456Z",
            "level":     "INFO",
            "event":     "bulk_test_completed",
            "details": {
                "model":    "Nagaraju_Final_ResNet.pth",
                "accuracy": 94.3,
                "total":    500
            }
        }

    Parameters
    ----------
    event_type:
        Short snake_case string identifying the event category, e.g.
        ``"model_loaded"``, ``"inference_error"``, ``"user_feedback_saved"``.
    details:
        Arbitrary key-value pairs to embed in the ``"details"`` field.
        Values must be JSON-serialisable; non-serialisable objects are
        coerced to their ``str()`` representation.
    level:
        Log severity label (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``).  Does **not** affect Python's logging level – it is
        recorded as a field in the JSON record only.
    feedback_root:
        Root directory under which ``events.log`` is written.

    Notes
    -----
    - This function never raises; all exceptions are caught and emitted via
      the module logger so that a logging failure cannot crash the caller.
    - The log file is opened in append mode; concurrent writes from multiple
      processes are *not* protected by a file lock.  For high-throughput
      scenarios consider replacing the append with a proper logging handler.
    """
    global EVENT_LOG_FILE  # noqa: PLW0603 – allow root override

    try:
        fb_root = Path(feedback_root)
        event_log = fb_root / "events.log"

        # Sanitise details: coerce non-serialisable values to str
        safe_details: dict[str, Any] = {}
        for k, v in (details or {}).items():
            try:
                json.dumps(v)
                safe_details[k] = v
            except (TypeError, ValueError):
                safe_details[k] = str(v)

        payload: dict[str, Any] = {
            "timestamp": _utc_now(),
            "level": level.upper(),
            "event": event_type,
            "details": safe_details,
        }

        _ensure_dir(fb_root)
        with event_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

        # Mirror to Python logger at the matching severity
        _log_fn = {
            "DEBUG": logger.debug,
            "INFO": logger.info,
            "WARNING": logger.warning,
            "ERROR": logger.error,
        }.get(level.upper(), logger.info)

        _log_fn("event=%s  details=%s", event_type, safe_details)

    except Exception:  # pragma: no cover
        # Last-resort: emit to stderr via the root logger and swallow
        logger.error(
            "log_event: failed to write event '%s':\n%s",
            event_type,
            traceback.format_exc(),
        )