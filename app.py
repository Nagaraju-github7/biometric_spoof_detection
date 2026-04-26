# app.py
"""
Advanced Fingerprint Spoof Detection — Streamlit UI
====================================================
Orchestration layer only.  All heavy logic lives in:

    models.py    – nn.Module classes + get_model_by_name()
    processor.py – A600 preprocessing + inference pipeline
    utils.py     – flatten_dataset(), save_user_feedback(), log_event()
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------
from models import get_model_by_name
from processor import predict_spoof_live, preprocess_a600_image, run_inference
from utils import (
    LABEL_NAME,
    flatten_dataset,
    log_event,
    save_user_feedback,
)

# ---------------------------------------------------------------------------
# Optional scanner module
# ---------------------------------------------------------------------------
try:
    from fingerprint_scanner import scanner_ui, FingerprintScanner
    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Fingerprint Spoof Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .danger-box {
        background: #ef4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .warning-box {
        background: #f59e0b;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .info-box {
        background: #3b82f6;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 8px;
        margin: 0.25rem 0;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers — model discovery
# ---------------------------------------------------------------------------

def get_available_models() -> list[str]:
    """Return .pth paths from ./models/, optimised models first."""
    models_dir = Path("./models")
    if not models_dir.exists():
        return []
    all_paths = [str(p) for p in models_dir.glob("*.pth")]
    optimised = [p for p in all_paths if "Nagaraju_Final_ResNet" in p]
    others = [p for p in all_paths if "Nagaraju_Final_ResNet" not in p]
    return optimised + others


# ---------------------------------------------------------------------------
# Helpers — dataset structure
# ---------------------------------------------------------------------------

def get_dataset_structure(folder: str | Path) -> dict:
    """
    Return a nested dict describing the folder layout and image counts.

    Example output for a material-based layout::

        {
            "Live":  150,
            "Fake":  {"Ecoflex": 80, "Gelatin": 70}
        }
    """
    root = Path(folder)
    if not root.exists():
        return {}

    structure: dict = {}
    subdirs = [d for d in root.iterdir() if d.is_dir()]

    if not subdirs:
        structure["Test Images"] = len(
            [f for f in root.iterdir()
             if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        )
        return structure

    for material_dir in subdirs:
        child_dirs = [d for d in material_dir.iterdir() if d.is_dir()]
        if child_dirs:
            structure[material_dir.name] = {
                sub.name: len(list(sub.glob("*.*"))) for sub in child_dirs
            }
        else:
            structure[material_dir.name] = len(list(material_dir.glob("*.*")))

    return structure


def display_structure(structure: dict, indent: int = 0) -> None:
    """Render a nested structure dict as indented Streamlit text."""
    for key, value in structure.items():
        if isinstance(value, dict):
            st.write("  " * indent + f"📁 {key}/")
            display_structure(value, indent + 1)
        else:
            st.write("  " * indent + f"📁 {key}/: {value} images")


# ---------------------------------------------------------------------------
# Helpers — bulk testing
# ---------------------------------------------------------------------------

def _collect_test_files(root: Path) -> list[dict]:
    """
    Walk *root* and return records: {path, material, true_label}.
    Live folders → true_label=1 ; everything else → true_label=0.
    """
    records = []
    subdirs = [d for d in root.iterdir() if d.is_dir()]

    if not subdirs:
        # Flat layout – no ground truth
        for f in root.iterdir():
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                records.append({"path": f, "material": "Test Images", "true_label": -1})
        return records

    for material_dir in subdirs:
        true_label = 1 if material_dir.name.lower() == "live" else 0
        child_dirs = [d for d in material_dir.iterdir() if d.is_dir()]

        if child_dirs:
            for sub in child_dirs:
                for f in sub.iterdir():
                    if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                        records.append(
                            {"path": f, "material": sub.name, "true_label": true_label}
                        )
        else:
            for f in material_dir.iterdir():
                if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    records.append(
                        {"path": f, "material": material_dir.name, "true_label": true_label}
                    )

    return records


def run_bulk_test(
    model: torch.nn.Module,
    device: torch.device,
    dataset_folder: str,
    model_name: str,
    progress_bar,
) -> list[dict]:
    """
    Iterate over every image in *dataset_folder*, run inference via
    processor.predict_spoof_live(), and return a flat results list.
    Failed images are copied to ./failures/<model_name>/<material>/.
    """
    root = Path(dataset_folder)
    records = _collect_test_files(root)
    if not records:
        st.error(f"❌ No images found in {dataset_folder}")
        return []

    results = []
    failures_root = Path("./failures") / model_name

    for i, rec in enumerate(records):
        img_path: Path = rec["path"]
        material: str = rec["material"]
        true_label: int = rec["true_label"]

        try:
            with Image.open(img_path) as pil_img:
                pil_copy = pil_img.copy()

            result = predict_spoof_live(model, device, pil_copy)
            predicted = result.class_index
            confidence = result.confidence
            is_correct = (predicted == true_label) if true_label != -1 else None

            row = {
                "image": str(img_path),
                "material": material,
                "true_label": true_label,
                "predicted_label": predicted,
                "confidence": confidence,
                "correct": is_correct,
            }
            results.append(row)

            # Copy failures for offline review
            if is_correct is False:
                dest_dir = failures_root / material
                dest_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(img_path, dest_dir / img_path.name)

        except Exception as exc:
            st.warning(f"⚠️ Skipped {img_path.name}: {exc}")
            log_event("bulk_test_image_error",
                      {"file": str(img_path), "error": str(exc)},
                      level="WARNING")

        progress_bar.progress((i + 1) / len(records))

    log_event("bulk_test_completed", {
        "model": model_name,
        "dataset": dataset_folder,
        "total": len(results),
    })
    return results


# ---------------------------------------------------------------------------
# Helpers — metrics & plots
# ---------------------------------------------------------------------------

def calculate_metrics(results: list[dict]) -> dict:
    """Compute overall and per-material accuracy from a results list."""
    if not results:
        return {}

    total = len(results)
    has_gt = any(r["correct"] is not None for r in results)
    correct = sum(1 for r in results if r["correct"] is True)
    overall_acc = (correct / total * 100) if has_gt else 0.0

    material_results: dict[str, list] = defaultdict(list)
    for r in results:
        material_results[r["material"]].append(r)

    material_metrics: dict[str, dict] = {}
    for mat, rows in material_results.items():
        mat_total = len(rows)
        mat_correct = sum(1 for r in rows if r["correct"] is True)
        mat_failures = mat_total - mat_correct if has_gt else 0
        material_metrics[mat] = {
            "accuracy": (mat_correct / mat_total * 100) if mat_total and has_gt else 0.0,
            "total": mat_total,
            "correct": mat_correct,
            "failures": mat_failures,
        }

    top_offenders = sorted(
        material_metrics.items(),
        key=lambda x: x[1]["failures"],
        reverse=True,
    )

    return {
        "overall_accuracy": overall_acc,
        "total_images": total,
        "correct_predictions": correct,
        "material_metrics": material_metrics,
        "top_offenders": top_offenders,
        "has_ground_truth": has_gt,
    }


def plot_confusion_matrix(results: list[dict]):
    """Return a matplotlib Figure for the confusion matrix (or distribution)."""
    if not results:
        return None

    has_gt = any(r["correct"] is not None for r in results)

    if not has_gt:
        from collections import Counter
        counts = Counter(r["predicted_label"] for r in results)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(["Live (1)", "Spoof (0)"], [counts.get(1, 0), counts.get(0, 0)],
               color=["green", "red"])
        ax.set_title("Prediction Distribution (no ground truth)")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Count")
        return fig

    true_labels = [r["true_label"] for r in results]
    pred_labels = [r["predicted_label"] for r in results]
    labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fake (0)", "Live (1)"],
        yticklabels=["Fake (0)", "Live (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    total = len(results)
    acc = sum(1 for r in results if r["correct"]) / total * 100
    ax.text(0.02, 0.98, f"Total: {total}\nAccuracy: {acc:.1f}%",
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    return fig


# ---------------------------------------------------------------------------
# UI helpers — shared result display
# ---------------------------------------------------------------------------

def _render_prediction(result, model, device, image: Image.Image) -> str:
    """
    Display class probabilities + verdict box.
    Returns "Live" or "Spoof" string for downstream feedback use.
    """
    tensor = preprocess_a600_image(image)
    probs, _ = run_inference(model, device, tensor)
    prob_live = float(probs[1].item())
    prob_spoof = float(probs[0].item())

    st.write("**Class Probabilities:**")
    st.write(f"🟢 Live  (Class 1): {prob_live:.3f}  ({prob_live*100:.1f}%)")
    st.write(f"🔴 Spoof (Class 0): {prob_spoof:.3f}  ({prob_spoof*100:.1f}%)")

    if result.class_index == 1:
        st.markdown('<div class="success-box">🔍 RESULT: LIVE FINGERPRINT</div>',
                    unsafe_allow_html=True)
        label_str = "Live"
    else:
        st.markdown('<div class="danger-box">⚠️ RESULT: SPOOF DETECTED</div>',
                    unsafe_allow_html=True)
        label_str = "Spoof"

    st.markdown(
        f'<div class="info-box">🎯 Confidence: {result.confidence:.1%}</div>',
        unsafe_allow_html=True,
    )
    return label_str


def _render_feedback_form(
    image: Image.Image,
    filename: str,
    predicted_label: int,
    confidence: float,
    model_result: str,
    key_suffix: str,
) -> None:
    """Render user-correction form and persist feedback via utils."""
    st.markdown("### 🔄 User Correction")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        user_label_str = st.radio(
            "Correct label:", ["Live", "Spoof"], key=f"label_{key_suffix}"
        )
        material_type = st.selectbox(
            "Material type (if spoof):",
            ["Silicone", "Gelatin", "Play-Doh", "Ecoflex",
             "Latex", "Body Double", "Other"],
            key=f"mat_{key_suffix}",
        )
    with col_b:
        if st.button("Submit Feedback", key=f"submit_{key_suffix}"):
            correct_label_int = 1 if user_label_str == "Live" else 0
            mat = material_type if user_label_str == "Spoof" else None
            ok, detail = save_user_feedback(
                image=image,
                original_filename=filename,
                predicted_label=predicted_label,
                correct_label=correct_label_int,
                confidence=confidence,
                material_type=mat,
            )
            if ok:
                st.success(f"✅ Saved → {Path(detail).name}")
                st.session_state.user_corrections.append({
                    "image": filename,
                    "model_prediction": model_result,
                    "user_label": user_label_str,
                    "confidence": confidence,
                    "material_type": mat,
                    "saved_image_path": detail,
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                st.error(f"❌ {detail}")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Session state initialisation ────────────────────────────────────────
    defaults = {
        "history": [],
        "stats": {"total_inferences": 0, "live_detections": 0,
                  "spoof_detections": 0, "accuracy": 0.0},
        "user_corrections": [],
        "feedback_mode": False,
        "bulk_results": None,
        "metrics": {},
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-header">'
        "🔐 Advanced Fingerprint Spoof Detection with User Feedback"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔧 Configuration")

    feedback_mode = st.sidebar.checkbox(
        "🔄 Enable User Feedback Mode",
        value=st.session_state.feedback_mode,
        help="Save user corrections and misclassified images for retraining.",
    )
    st.session_state.feedback_mode = feedback_mode

    if feedback_mode:
        st.sidebar.success("✅ Feedback mode enabled")
    else:
        st.sidebar.info("📝 Feedback mode disabled")

    # ── Model selection ──────────────────────────────────────────────────────
    available_models = get_available_models()
    if not available_models:
        st.error("❌ No .pth files found in ./models/")
        st.stop()

    model_labels = [
        f"🚀 {Path(p).name} (Optimised)" if "Nagaraju_Final_ResNet" in p
        else f"📊 {Path(p).name} (Legacy)"
        for p in available_models
    ]

    selected_idx = st.sidebar.selectbox(
        "Select Model:",
        range(len(model_labels)),
        format_func=lambda x: model_labels[x],
        index=0,
    )
    selected_model_path = available_models[selected_idx]
    model_name = Path(selected_model_path).stem

    if "Nagaraju_Final_ResNet" in selected_model_path:
        st.sidebar.success(f"🚀 Optimised: {model_name}")
    else:
        st.sidebar.info(f"📊 Legacy: {model_name}")

    # ── Load model (cached) ──────────────────────────────────────────────────
    @st.cache_resource
    def _load(path: str):
        return get_model_by_name(path)

    try:
        model, device = _load(selected_model_path)
    except Exception as exc:
        st.sidebar.error(f"❌ Model load failed: {exc}")
        log_event("model_load_error",
                  {"path": selected_model_path, "error": str(exc)},
                  level="ERROR")
        st.stop()

    st.sidebar.success(f"✅ Model loaded")
    st.sidebar.info(f"🖥️ Device: {device}")
    log_event("model_loaded", {"model": model_name, "device": str(device)})

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_names = ["🖼️ Single Test", "📊 Bulk Test",
                 "📈 Analysis", "🔄 User Feedback"]
    if SCANNER_AVAILABLE:
        tab_names.append("🔌 Scanner Input")

    tabs = st.tabs(tab_names)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 – Single image test
    # ════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### 🖼️ Single Image Testing")

        uploaded = st.file_uploader(
            "Upload fingerprint image:",
            type=["png", "jpg", "jpeg", "bmp"],
            key="single_upload",
        )

        if uploaded:
            with Image.open(uploaded) as raw:
                image = raw.copy()

            col_img, col_res = st.columns(2)

            with col_img:
                st.image(image, caption="📷 Original Image", use_column_width=True)

                # Preprocessed preview
                try:
                    t = preprocess_a600_image(image)
                    preview = t.permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    preview = np.clip(preview * std + mean, 0, 1)
                    st.image(preview, caption="🔧 Preprocessed (224×224 crop)",
                             use_column_width=True)
                except Exception as exc:
                    st.warning(f"Preview unavailable: {exc}")

            with col_res:
                st.markdown("### 📊 Prediction Results")
                try:
                    result = predict_spoof_live(model, device, image)
                    model_result = _render_prediction(result, model, device, image)

                    st.session_state.stats["total_inferences"] += 1
                    if result.class_index == 1:
                        st.session_state.stats["live_detections"] += 1
                    else:
                        st.session_state.stats["spoof_detections"] += 1

                    log_event("single_inference", {
                        "file": uploaded.name,
                        "predicted": model_result,
                        "confidence": round(result.confidence, 4),
                        "inference_ms": round(result.inference_ms, 1),
                    })

                    if st.session_state.feedback_mode:
                        _render_feedback_form(
                            image=image,
                            filename=uploaded.name,
                            predicted_label=result.class_index,
                            confidence=result.confidence,
                            model_result=model_result,
                            key_suffix=f"single_{uploaded.name}",
                        )
                except Exception as exc:
                    st.error(f"❌ Inference failed: {exc}")
                    log_event("inference_error",
                              {"file": uploaded.name, "error": str(exc)},
                              level="ERROR")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 – Bulk test
    # ════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### 📊 Bulk Dataset Testing")
        st.markdown("#### 📁 Dataset Folder")

        col_path, col_info = st.columns([2, 1])

        with col_path:
            dataset_folder = st.text_input(
                "Dataset folder path:",
                value="./dataset/test",
                key="bulk_folder",
            )

            qc1, qc2, qc3 = st.columns(3)
            with qc2:
                if st.button("📂 Browse Help", key="browse_help"):
                    st.info(
                        "Copy the full path from your file explorer and paste above.\n\n"
                        "Examples:\n"
                        "- `./dataset/test`\n"
                        "- `C:/data/fingerprint_test`\n"
                        "- `../other_project/test_data`"
                    )

        with col_info:
            test_path = Path(dataset_folder)
            if test_path.exists():
                total_files = len(list(test_path.rglob("*.*")))
                subdirs_count = len([d for d in test_path.iterdir() if d.is_dir()])
                st.success("✅ Folder found")
                st.write(f"📊 {total_files} files, {subdirs_count} subfolders")
            else:
                st.error("❌ Folder not found")
                st.stop()

        # Dataset structure preview
        ds_structure = get_dataset_structure(dataset_folder)
        if ds_structure:
            st.write("**Dataset Structure:**")
            display_structure(ds_structure)
        else:
            st.error("❌ Empty or unreadable dataset folder.")
            st.stop()

        # Flat-layout material override
        has_subdirs = any(Path(dataset_folder).iterdir()) and any(
            d.is_dir() for d in Path(dataset_folder).iterdir()
        )
        material_override = None
        if not has_subdirs:
            st.markdown("#### 🎯 Material Selection (flat layout)")
            material_override = st.selectbox(
                "Assign material type to all images:",
                ["Live Fingerprint", "Silicone", "Gelatin", "Play-Doh",
                 "Ecoflex", "Latex", "Body Double", "Unknown"],
                key="flat_material",
            )

        if st.button("🚀 Run Full Dataset Test", type="primary", key="run_bulk"):
            progress_bar = st.progress(0)
            with st.spinner("Running bulk test…"):
                results = run_bulk_test(
                    model=model,
                    device=device,
                    dataset_folder=dataset_folder,
                    model_name=model_name,
                    progress_bar=progress_bar,
                )
            if results:
                metrics = calculate_metrics(results)
                st.session_state.bulk_results = results
                st.session_state.metrics = metrics
                if metrics.get("has_ground_truth"):
                    st.success(
                        f"✅ Done! Accuracy: {metrics['overall_accuracy']:.1f}% "
                        f"over {metrics['total_images']} images."
                    )
                else:
                    st.success(
                        f"✅ Done! Processed {metrics['total_images']} images "
                        "(no ground truth → accuracy not computed)."
                    )
            else:
                st.error("❌ Bulk test returned no results.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 – Analysis
    # ════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### 📈 Analysis Results")

        if st.session_state.bulk_results is None:
            st.info("Run a Bulk Test (Tab 2) first to see analysis here.")
            st.stop()

        results = st.session_state.bulk_results
        metrics = st.session_state.metrics

        # ── Overall KPIs ─────────────────────────────────────────────────
        st.markdown("#### 📊 Overall Performance")
        k1, k2, k3 = st.columns(3)

        with k1:
            if metrics["has_ground_truth"]:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>{metrics["overall_accuracy"]:.1f}%</h3>'
                    f'<p>Overall Accuracy</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="warning-box"><h3>N/A</h3>'
                    "<p>No Ground Truth</p></div>",
                    unsafe_allow_html=True,
                )

        with k2:
            st.markdown(
                f'<div class="metric-card">'
                f'<h3>{metrics["total_images"]}</h3>'
                f"<p>Total Images</p></div>",
                unsafe_allow_html=True,
            )

        with k3:
            val = metrics["correct_predictions"] if metrics["has_ground_truth"] else "—"
            st.markdown(
                f'<div class="metric-card">'
                f"<h3>{val}</h3>"
                f"<p>Correct Predictions</p></div>",
                unsafe_allow_html=True,
            )

        # ── Per-material table ───────────────────────────────────────────
        st.markdown("#### 🎯 Per-Material Performance")
        mat_rows = []
        for mat, m in metrics["material_metrics"].items():
            status = "🟢" if m["total"] > 0 else "🔴"
            mat_rows.append({
                "Material": f"{status} {mat}",
                "Accuracy": f"{m['accuracy']:.1f}%" if m["total"] > 0 else "N/A",
                "Total": m["total"],
                "Failures": m["failures"],
            })
        st.dataframe(pd.DataFrame(mat_rows), use_container_width=True)

        # ── Top offenders ────────────────────────────────────────────────
        st.markdown("#### ⚠️ Top Offenders")
        valid = [(mat, m) for mat, m in metrics["top_offenders"] if m["total"] > 0]
        if valid:
            for i, (mat, m) in enumerate(valid[:5]):
                rate = m["failures"] / m["total"] * 100
                st.write(
                    f"{i+1}. **{mat}**: {m['failures']}/{m['total']} "
                    f"failures ({rate:.1f}%)"
                )
        else:
            st.info("No failures detected or all folders are empty.")

        # ── Confusion matrix ─────────────────────────────────────────────
        st.markdown("#### 📈 Confusion Matrix")
        fig = plot_confusion_matrix(results)
        if fig:
            st.pyplot(fig)

        # ── Export ───────────────────────────────────────────────────────
        st.markdown("#### 💾 Export Results")
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        ec1, ec2 = st.columns(2)

        with ec1:
            csv = pd.DataFrame(results).to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                data=csv,
                file_name=f"results_{model_name}_{ts_str}.csv",
                mime="text/csv",
            )
        with ec2:
            import json
            # Convert Path objects to strings for JSON serialisation
            safe_metrics = {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in metrics.items()
            }
            st.download_button(
                "📥 Download Metrics JSON",
                data=json.dumps(safe_metrics, indent=2),
                file_name=f"metrics_{model_name}_{ts_str}.json",
                mime="application/json",
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 – User Feedback
    # ════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### 🔄 User Feedback History")

        if not st.session_state.feedback_mode:
            st.info("Enable Feedback Mode in the sidebar to record corrections.")
            st.stop()

        feedback_dir = Path("./user_feedback")

        # ── Directory tree ───────────────────────────────────────────────
        if feedback_dir.exists():
            st.markdown("#### 📁 Feedback Directory")

            def _tree(path: Path, indent: int = 0) -> None:
                icon = "📁" if path.is_dir() else "📄"
                st.write("  " * indent + f"{icon} {path.name}")
                if path.is_dir():
                    for child in sorted(path.iterdir()):
                        _tree(child, indent + 1)

            _tree(feedback_dir)

            live_count = len(list(feedback_dir.glob("Live/**/*.png")))
            spoof_count = len(list(feedback_dir.glob("Spoof/**/*.png")))
            total_fb = live_count + spoof_count

            st.markdown("#### 📊 Feedback Statistics")
            fb1, fb2, fb3 = st.columns(3)
            for col, label, val in zip(
                [fb1, fb2, fb3],
                ["Live Images", "Spoof Images", "Total Images"],
                [live_count, spoof_count, total_fb],
            ):
                col.markdown(
                    f'<div class="metric-card"><h3>{val}</h3>'
                    f"<p>{label}</p></div>",
                    unsafe_allow_html=True,
                )

        # ── Corrections table ────────────────────────────────────────────
        if st.session_state.user_corrections:
            st.markdown("#### 📝 Corrections Log")
            st.dataframe(
                pd.DataFrame(st.session_state.user_corrections),
                use_container_width=True,
            )

            total_c = len(st.session_state.user_corrections)
            wrong_c = sum(
                1 for c in st.session_state.user_corrections
                if c["model_prediction"] != c["user_label"]
            )
            agree_rate = (total_c - wrong_c) / total_c * 100 if total_c else 0

            fb_a, fb_b, fb_c = st.columns(3)
            for col, label, val in zip(
                [fb_a, fb_b, fb_c],
                ["Total Corrections", "Model Was Wrong", "Agreement Rate"],
                [total_c, wrong_c, f"{agree_rate:.1f}%"],
            ):
                col.markdown(
                    f'<div class="metric-card"><h3>{val}</h3>'
                    f"<p>{label}</p></div>",
                    unsafe_allow_html=True,
                )

            if st.button("📥 Export Feedback CSV", key="export_fb"):
                fb_csv = pd.DataFrame(st.session_state.user_corrections).to_csv(
                    index=False
                )
                st.download_button(
                    "📥 Download",
                    data=fb_csv,
                    file_name=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        else:
            st.info("No corrections recorded yet.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 – Scanner Input  (only rendered when module is available)
    # ════════════════════════════════════════════════════════════════════════
    if SCANNER_AVAILABLE and len(tabs) == 5:
        with tabs[4]:
            st.markdown("### 🔌 Live Scanner Input")

            try:
                scanner = FingerprintScanner()
                st.success("✅ Scanner module loaded")
            except Exception as exc:
                st.error(f"❌ Scanner init failed: {exc}")
                scanner = None

            # ── Input method ─────────────────────────────────────────────
            input_method = st.radio(
                "Input method:",
                ["USB Camera", "Upload Image", "Test Mode"],
                key="scanner_input_method",
            )

            captured_image: Image.Image | None = None
            captured_name: str = "scanner_capture"

            # ── USB Camera ───────────────────────────────────────────────
            if input_method == "USB Camera":
                cam_idx = st.selectbox("Camera index:", [0, 1, 2, 3], index=0)
                c1, c2, c3 = st.columns(3)

                with c1:
                    if st.button("🔌 Connect", key="cam_connect"):
                        cap = cv2.VideoCapture(cam_idx)
                        if cap.isOpened():
                            st.session_state.camera_cap = cap
                            st.success(f"✅ Camera {cam_idx} connected")
                        else:
                            st.error("❌ Could not open camera")

                with c2:
                    if st.button("📸 Capture Frame", key="cam_capture"):
                        cap = st.session_state.get("camera_cap")
                        if cap and cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                st.session_state.scanner_image = Image.fromarray(rgb)
                                st.session_state.scanner_name = (
                                    f"camera_{_timestamp_slug()}.png"
                                    if True else "camera_capture.png"
                                )
                                st.success("✅ Frame captured")
                            else:
                                st.error("❌ Frame capture failed")
                        else:
                            st.error("❌ Connect a camera first")

                with c3:
                    if st.button("⏹️ Release Camera", key="cam_release"):
                        cap = st.session_state.pop("camera_cap", None)
                        if cap:
                            cap.release()
                        st.info("Camera released")

            # ── Upload ───────────────────────────────────────────────────
            elif input_method == "Upload Image":
                up = st.file_uploader(
                    "Upload fingerprint image:",
                    type=["png", "jpg", "jpeg", "bmp"],
                    key="scanner_upload",
                )
                if up:
                    with Image.open(up) as raw:
                        st.session_state.scanner_image = raw.copy()
                    st.session_state.scanner_name = up.name
                    st.success("✅ Image uploaded")

            # ── Test mode ────────────────────────────────────────────────
            elif input_method == "Test Mode":
                test_root = Path("./dataset/test")
                samples = (
                    list(test_root.rglob("*.png"))[:1]
                    if test_root.exists() else []
                )
                if samples:
                    with Image.open(samples[0]) as raw:
                        st.session_state.scanner_image = raw.copy()
                    st.session_state.scanner_name = samples[0].name
                    st.success(f"✅ Loaded test image: {samples[0].name}")
                else:
                    st.warning("⚠️ No test images found in ./dataset/test")

            # ── Analyse captured image ───────────────────────────────────
            scanner_img: Image.Image | None = st.session_state.get("scanner_image")
            scanner_name: str = st.session_state.get("scanner_name", "scanner_capture")

            if scanner_img is not None:
                st.markdown("#### 📸 Captured Image")
                col_si, col_sr = st.columns(2)

                with col_si:
                    st.image(scanner_img, width=400, caption="Captured Image")
                    if st.button("🔄 Analyse", key="scanner_analyse"):
                        try:
                            res = predict_spoof_live(model, device, scanner_img)
                            st.session_state.scanner_result = res
                            log_event("scanner_inference", {
                                "file": scanner_name,
                                "predicted": LABEL_NAME[res.class_index],
                                "confidence": round(res.confidence, 4),
                            })
                        except Exception as exc:
                            st.error(f"❌ {exc}")
                            log_event("scanner_inference_error",
                                      {"error": str(exc)}, level="ERROR")

                with col_sr:
                    res = st.session_state.get("scanner_result")
                    if res is not None:
                        st.markdown("### 📊 Results")
                        model_result = _render_prediction(
                            res, model, device, scanner_img
                        )
                        if st.session_state.feedback_mode:
                            _render_feedback_form(
                                image=scanner_img,
                                filename=scanner_name,
                                predicted_label=res.class_index,
                                confidence=res.confidence,
                                model_result=model_result,
                                key_suffix=f"scanner_{scanner_name}",
                            )

                if st.button("🗑️ Clear", key="scanner_clear"):
                    for k in ("scanner_image", "scanner_name", "scanner_result"):
                        st.session_state.pop(k, None)
                    st.rerun()


def _timestamp_slug() -> str:
    """Filesystem-safe UTC timestamp slug."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]


if __name__ == "__main__":
    main()