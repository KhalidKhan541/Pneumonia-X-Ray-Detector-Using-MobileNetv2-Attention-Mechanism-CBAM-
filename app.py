from __future__ import annotations

import base64
import io
import json
import os
import warnings
from datetime import datetime
from html import escape
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import keras
import numpy as np
import streamlit as st
import tensorflow as tf
from keras import layers
from PIL import Image, ImageOps

# ── Paths and constants ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "improved" / "mobilenet_improved.keras"
BASELINE_JSON = BASE_DIR / "models" / "baseline_results.json"
IMPROVED_JSON = BASE_DIR / "models" / "improved" / "improved_results.json"
REPORT_PDF = BASE_DIR / "Pneumonia_Detection_Report.pdf"
HERO_SCAN = BASE_DIR / "data" / "samples" / "hero_scan.jpg"
HERO_HEATMAP = BASE_DIR / "data" / "samples" / "hero_heatmap.jpg"

SAMPLES = {
    "normal": {
        "path": BASE_DIR / "data" / "samples" / "IM-0001-0001.jpeg",
        "label": "Sample A · normal",
        "short": "A · Normal",
        "truth": "NORMAL",
    },
    "pneumonia": {
        "path": BASE_DIR / "data" / "samples" / "person38_virus_83.jpeg",
        "label": "Sample B · pneumonia",
        "short": "B · Pneumonia",
        "truth": "PNEUMONIA",
    },
}

IMG_SIZE = 224
MODEL_LABEL = "MobileNetV2 + CBAM"
FALLBACK_BASELINE = {
    "accuracy": 0.8798,
    "roc_auc": 0.9582,
    "precision": 0.8723,
    "recall": 0.9462,
    "f1_score": 0.9077,
}
FALLBACK_IMPROVED = {
    "accuracy": 0.8814102564102564,
    "roc_auc": 0.9430089853166777,
    "precision": 0.8708920187793427,
    "recall": 0.9512820512820512,
    "f1_score": 0.9093137254901961,
}

st.set_page_config(
    page_title="PNEUMO/CXR — Pneumonia Detection",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore", message=".*build\\(\\) was called on layer.*")


# ── Keras custom layers (must match the training notebook) ─────────────────
@keras.saving.register_keras_serializable(package="pneumonia_cbam")
class ChannelAttention(layers.Layer):
    def __init__(self, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        self.dense1 = layers.Dense(channels // self.reduction, activation="relu")
        self.dense2 = layers.Dense(channels, activation="sigmoid")
        self.global_avg = layers.GlobalAveragePooling2D()
        self.global_max = layers.GlobalMaxPooling2D()

    def call(self, x):
        avg_pool = self.global_avg(x)
        max_pool = self.global_max(x)
        avg_out = self.dense2(self.dense1(avg_pool))
        max_out = self.dense2(self.dense1(max_pool))
        attention = layers.Add()([avg_out, max_out])
        attention = layers.Activation("sigmoid")(attention)
        attention = layers.Reshape((1, 1, -1))(attention)
        return layers.Multiply()([x, attention])


@keras.saving.register_keras_serializable(package="pneumonia_cbam")
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            1, self.kernel_size, padding="same", activation="sigmoid"
        )

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = layers.Concatenate()([avg_pool, max_pool])
        attention = self.conv(concat)
        return layers.Multiply()([x, attention])


@keras.saving.register_keras_serializable(package="pneumonia_cbam")
class CBAM(layers.Layer):
    def __init__(self, reduction: int = 16, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.channel_att = ChannelAttention(reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


CUSTOM_OBJECTS = {
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention,
    "CBAM": CBAM,
}


# ── Data / model helpers ────────────────────────────────────────────────────
def read_json(path: Path, fallback: dict) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {**fallback, **data}
    except Exception:
        return dict(fallback)


BASELINE = read_json(BASELINE_JSON, FALLBACK_BASELINE)
IMPROVED = read_json(IMPROVED_JSON, FALLBACK_IMPROVED)


@st.cache_resource(show_spinner=False)
def load_assets():
    model = keras.models.load_model(
        str(MODEL_PATH), compile=False, custom_objects=CUSTOM_OBJECTS
    )
    layer_names = {layer.name for layer in model.layers}
    target_layer = "cbam" if "cbam" in layer_names else next(
        (layer.name for layer in model.layers if "mobilenet" in layer.name), None
    )
    if target_layer is None:
        raise RuntimeError("Could not find a convolutional feature layer in the model.")
    grad_model = keras.Model(
        model.inputs, [model.get_layer(target_layer).output, model.output]
    )
    return model, grad_model, target_layer


@st.cache_data(show_spinner=False)
def image_data_uri(path_string: str) -> str:
    path = Path(path_string)
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def open_image(file_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (0, 0, 0))
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    else:
        image = image.convert("RGB")
    return image


def preprocess_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    resized = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0), resized


def predict_probability(model, model_input: np.ndarray) -> float:
    output = model(model_input, training=False)
    return float(np.asarray(output).reshape(-1)[0])


def make_gradcam(
    grad_model,
    model_input: np.ndarray,
    positive_class: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(model_input, training=False)
        probability = predictions[:, 0]
        score = probability if positive_class else 1.0 - probability

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients were unavailable.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = np.asarray(heatmap, dtype=np.float32)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    base = np.asarray(
        Image.fromarray((model_input[0] * 255).astype(np.uint8)).convert("L"),
        dtype=np.float32,
    ) / 255.0

    # Gamma concentrates the visual on the strongest attention peaks.
    display_heat = heatmap**2.0
    screen_overlay = 1.0 - (1.0 - base) * (1.0 - 0.72 * display_heat)
    screen_overlay = np.clip(screen_overlay, 0.0, 1.0)

    base_rgb = np.dstack([base, base, base])
    heat_rgb = np.dstack([display_heat, display_heat, display_heat])
    overlay_rgb = np.dstack([screen_overlay, screen_overlay, screen_overlay])
    return (
        (base_rgb * 255).astype(np.uint8),
        (heat_rgb * 255).astype(np.uint8),
        (overlay_rgb * 255).astype(np.uint8),
    )


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def bar_width(value: float, low: float = 0.80, high: float = 1.0) -> float:
    return float(np.clip((value - low) / (high - low) * 100.0, 4.0, 100.0))


def risk_band(probability: float) -> str:
    if probability >= 0.65:
        return "Elevated"
    if probability >= 0.35:
        return "Borderline"
    return "Low"


# ── Styling ────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --ink: #0b0b0b;
    --paper: #f4f3ef;
    --white: #ffffff;
    --muted: #6e6b64;
    --line: #d8d5cd;
    --soft: #e9e6df;
}

html, body, .stApp {
    background: var(--paper) !important;
    color: var(--ink);
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}

.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6 {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
#MainMenu,
footer {
    display: none !important;
}

[data-testid="stExpandSidebarButton"] {
    display: flex !important;
    align-items: center;
    justify-content: center;
    width: 2.1rem !important;
    height: 2.1rem !important;
    border: 1px solid var(--line) !important;
    background: var(--paper) !important;
    color: var(--ink) !important;
    border-radius: 0 !important;
    transition: background 160ms ease, color 160ms ease;
}

[data-testid="stExpandSidebarButton"]:hover {
    background: var(--ink) !important;
    color: #ffffff !important;
}

[data-testid="stExpandSidebarButton"] span,
[data-testid="stExpandSidebarButton"] svg {
    color: inherit !important;
}

[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
}

[data-testid="stSidebarCollapseButton"] button {
    display: flex !important;
    align-items: center;
    justify-content: center;
    width: 2.1rem !important;
    height: 2.1rem !important;
    border: 1px solid #4a4843 !important;
    background: transparent !important;
    color: #f3f1ea !important;
    border-radius: 0 !important;
    transition: background 160ms ease, color 160ms ease;
}

[data-testid="stSidebarCollapseButton"] button:hover {
    background: #ffffff !important;
    color: var(--ink) !important;
}

[data-testid="stAppViewBlockContainer"] {
    max-width: 1420px;
    padding: 1.4rem 3.4rem 0 3.4rem;
}

[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: 1px solid #222222;
    width: 318px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.6rem 1.45rem 2.4rem 1.45rem;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #f3f1ea;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #9d9a92;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    line-height: 1.5;
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button {
    background: transparent !important;
    color: #f3f1ea !important;
    border: 1px solid #4a4843 !important;
    border-radius: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: #ffffff !important;
    color: var(--ink) !important;
    border-color: #ffffff !important;
}

[data-testid="stSidebar"] hr {
    border: none;
    border-top: 1px solid #2b2a27;
    margin: 1.35rem 0;
}

[data-testid="stSidebarCollapsedControl"] {
    color: var(--ink);
}

.stButton > button, .stDownloadButton > button {
    border-radius: 0 !important;
    border: 1px solid var(--ink) !important;
    box-shadow: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    min-height: 2.75rem !important;
    transition: all 160ms ease !important;
}

button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: var(--ink) !important;
    color: #ffffff !important;
}

button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background: transparent !important;
    color: var(--ink) !important;
}

button[kind="secondary"],
button[data-testid="stBaseButton-secondary"] {
    background: var(--white) !important;
    color: var(--ink) !important;
}

button[kind="secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    background: var(--ink) !important;
    color: #ffffff !important;
}

button[kind="tertiary"],
button[data-testid="stBaseButton-tertiary"] {
    background: transparent !important;
    border-color: transparent !important;
    color: var(--muted) !important;
    text-decoration: underline;
    text-underline-offset: 4px;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--white);
    border: 1px dashed #a7a39a;
    border-radius: 0;
    padding: 1.45rem 1.2rem;
    transition: border-color 160ms ease, background 160ms ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--ink);
    background: #fbfaf7;
}

[data-testid="stFileUploaderDropzone"] button {
    background: var(--ink) !important;
    color: #ffffff !important;
    border-radius: 0 !important;
    border: 1px solid var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.64rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: transparent !important;
    color: var(--ink) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    color: var(--ink) !important;
    font-size: 0.86rem !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
}

[data-testid="stFileUploaderFile"] {
    background: var(--white);
    border: 1px solid var(--line);
    border-radius: 0;
}

[data-testid="stSlider"] label p {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--muted);
}

[data-testid="stCaptionContainer"] p {
    color: var(--muted);
    font-size: 0.74rem;
    line-height: 1.55;
}

[data-testid="stAlert"] {
    border-radius: 0;
    border: 1px solid var(--ink);
    background: var(--white);
    color: var(--ink);
}

[data-testid="stAlert"] svg {
    color: var(--ink);
}

[data-testid="stImage"] img {
    border: 1px solid var(--line);
    display: block;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--white);
    border: 1px solid var(--line) !important;
    border-radius: 0 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 1.2rem 1.2rem 1.25rem 1.2rem;
}

::selection {
    background: var(--ink);
    color: #ffffff;
}

/* ── Top navigation ───────────────────────────────────────────────────── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    border-bottom: 1px solid var(--ink);
    padding: 0.65rem 0 0.8rem 0;
    margin-bottom: 2.6rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    white-space: nowrap;
}

.brand-mark {
    width: 23px;
    height: 23px;
    display: grid;
    place-items: center;
    background: var(--ink);
    color: #ffffff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    line-height: 1;
}

.brand .slash {
    color: var(--muted);
    font-weight: 300;
    margin: 0 0.05rem;
}

.topnav {
    display: flex;
    align-items: center;
    gap: 1.6rem;
}

.topnav a {
    color: var(--muted);
    text-decoration: none;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    transition: color 160ms ease;
}

.topnav a:hover {
    color: var(--ink);
}

.top-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.status-dot {
    width: 6px;
    height: 6px;
    display: inline-block;
    border-radius: 50%;
    background: var(--ink);
}

/* ── Hero ─────────────────────────────────────────────────────────────── */
.hero {
    display: grid;
    grid-template-columns: 1.04fr 0.96fr;
    gap: 3.4rem;
    align-items: stretch;
    margin: 0.8rem 0 4.6rem 0;
}

.hero-copy {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 0.8rem 0;
}

.eyebrow {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.eyebrow::before {
    content: "";
    width: 30px;
    height: 1px;
    background: var(--ink);
    flex: 0 0 auto;
}

.hero h1 {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-size: clamp(3.8rem, 6.7vw, 6.2rem) !important;
    font-weight: 400 !important;
    line-height: 0.9 !important;
    letter-spacing: -0.045em !important;
    color: var(--ink) !important;
    margin: 1.45rem 0 1.5rem 0 !important;
}

.hero h1 em {
    font-style: italic;
    font-weight: 400;
}

.hero-lede {
    max-width: 520px;
    color: #3e3c37;
    font-size: 0.98rem;
    font-weight: 300;
    line-height: 1.7;
    margin: 0;
}

.hero-actions {
    display: flex;
    align-items: center;
    gap: 1.15rem;
    flex-wrap: wrap;
    margin-top: 1.8rem;
}

.btn-primary {
    display: inline-flex;
    align-items: center;
    gap: 0.8rem;
    background: var(--ink);
    color: #ffffff !important;
    border: 1px solid var(--ink);
    padding: 0.85rem 1.15rem;
    text-decoration: none !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    transition: all 160ms ease;
}

.btn-primary:hover {
    background: transparent;
    color: var(--ink) !important;
}

.hero-note {
    max-width: 220px;
    color: var(--muted);
    font-size: 0.7rem;
    line-height: 1.5;
}

.hero-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    max-width: 550px;
    border-top: 1px solid var(--line);
    margin-top: 2.3rem;
    padding-top: 1.1rem;
}

.hero-stats > div {
    padding-right: 1rem;
    border-right: 1px solid var(--line);
}

.hero-stats > div:last-child {
    border-right: none;
}

.hero-stats span {
    display: block;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.56rem;
    letter-spacing: 0.11em;
    margin-bottom: 0.4rem;
}

.hero-stats b {
    display: block;
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.72rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    line-height: 1;
    color: var(--ink);
}

.scan-card {
    position: relative;
    min-height: 478px;
    overflow: hidden;
    background: #000000;
    border: 1px solid var(--ink);
}

.scan-card img.scan {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    filter: grayscale(1) contrast(1.08);
    opacity: 0.92;
}

.scan-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(180deg, rgba(0,0,0,0.48) 0%, transparent 28%, transparent 55%, rgba(0,0,0,0.86) 100%);
    pointer-events: none;
}

.scan-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.055) 1px, transparent 1px);
    background-size: 46px 46px;
    pointer-events: none;
}

.scan-top {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    z-index: 3;
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.9rem 1rem;
    color: #ffffff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.57rem;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.scan-top span {
    background: rgba(0,0,0,0.35);
    padding: 0.2rem 0.32rem;
}

.corner {
    position: absolute;
    z-index: 3;
    width: 13px;
    height: 13px;
    border-color: rgba(255,255,255,0.62);
    border-style: solid;
    border-width: 0;
}

.corner.tl { top: 9px; left: 9px; border-top-width: 1px; border-left-width: 1px; }
.corner.tr { top: 9px; right: 9px; border-top-width: 1px; border-right-width: 1px; }
.corner.bl { bottom: 9px; left: 9px; border-bottom-width: 1px; border-left-width: 1px; }
.corner.br { bottom: 9px; right: 9px; border-bottom-width: 1px; border-right-width: 1px; }

.crosshair-h {
    position: absolute;
    z-index: 2;
    top: 50%;
    left: 50%;
    width: 154px;
    height: 1px;
    background: rgba(255,255,255,0.24);
    transform: translate(-50%, -50%);
}

.crosshair-v {
    position: absolute;
    z-index: 2;
    top: 50%;
    left: 50%;
    width: 1px;
    height: 154px;
    background: rgba(255,255,255,0.24);
    transform: translate(-50%, -50%);
}

.scan-bottom {
    position: absolute;
    z-index: 4;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    padding: 1.1rem;
}

.scan-pred-label {
    color: rgba(255,255,255,0.62);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.scan-pred {
    color: #ffffff;
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.8rem;
    line-height: 1;
    margin-top: 0.4rem;
    letter-spacing: -0.02em;
}

.scan-pred b {
    display: inline-block;
    margin-left: 0.42rem;
    padding: 0.18rem 0.35rem;
    background: #ffffff;
    color: #000000;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    vertical-align: 0.18rem;
}

.heat-inset {
    flex: 0 0 106px;
    width: 106px;
    background: #000000;
    border: 1px solid rgba(255,255,255,0.58);
}

.heat-inset img {
    display: block;
    width: 100%;
    height: 91px;
    object-fit: cover;
    filter: grayscale(1);
}

.heat-inset span {
    display: block;
    border-top: 1px solid rgba(255,255,255,0.22);
    padding: 0.25rem 0.3rem;
    color: rgba(255,255,255,0.7);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

/* ── Section headings ─────────────────────────────────────────────────── */
.section-head {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.5rem;
    border-top: 1px solid var(--ink);
    padding-top: 0.95rem;
    margin: 3.7rem 0 1.55rem 0;
    scroll-margin-top: 1.2rem;
}

.section-id {
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.section-head h2 {
    margin: 0;
    color: var(--ink);
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-size: 2.6rem !important;
    font-weight: 400 !important;
    line-height: 0.98 !important;
    letter-spacing: -0.025em !important;
}

.section-head > p {
    max-width: 355px;
    margin: 0 0 0.2rem 0;
    color: var(--muted);
    font-size: 0.76rem;
    line-height: 1.6;
    text-align: right;
}

/* ── Inspect panel / results ──────────────────────────────────────────── */
.panel-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.57rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
}

.panel-label b {
    color: var(--ink);
    font-weight: 500;
}

.guidance {
    margin-top: 1.15rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.7rem;
    line-height: 1.6;
}

.guidance span {
    display: block;
    color: var(--ink);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.sample-note {
    margin-top: 0.5rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.56rem;
    line-height: 1.5;
    letter-spacing: 0.02em;
}

.empty-state {
    display: flex;
    min-height: 494px;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2.2rem;
    text-align: center;
    background: var(--white);
    border: 1px dashed var(--line);
}

.empty-icon {
    display: grid;
    width: 72px;
    height: 72px;
    place-items: center;
    border: 1px solid var(--ink);
    margin-bottom: 1.2rem;
    color: var(--ink);
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 2.2rem;
    line-height: 1;
}

.empty-kicker {
    margin-bottom: 0.45rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.56rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.empty-state h3 {
    margin: 0 0 0.75rem 0;
    color: var(--ink);
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-size: 2.2rem !important;
    font-weight: 400 !important;
    line-height: 0.95 !important;
    letter-spacing: -0.03em !important;
}

.empty-state h3 em {
    font-style: italic;
}

.empty-state p {
    max-width: 330px;
    margin: 0;
    color: var(--muted);
    font-size: 0.8rem;
    line-height: 1.65;
}

.empty-meta {
    margin-top: 1.55rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.54rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.verdict {
    padding: 1.35rem 1.4rem 0 1.4rem;
    border: 1px solid var(--ink);
}

.verdict-positive {
    background: var(--ink);
    color: #ffffff;
}

.verdict-negative {
    background: var(--white);
    color: var(--ink);
}

.verdict-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.22);
    padding-bottom: 0.75rem;
    color: rgba(255,255,255,0.6);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.verdict-negative .verdict-head {
    border-bottom-color: var(--line);
    color: var(--muted);
}

.verdict-body {
    padding: 1.25rem 0 1.35rem 0;
}

.verdict-kicker {
    color: rgba(255,255,255,0.58);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

.verdict-negative .verdict-kicker {
    color: var(--muted);
}

.verdict h3 {
    margin: 0 0 0.8rem 0;
    color: #ffffff;
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-size: clamp(2.5rem, 4.5vw, 3.9rem) !important;
    font-weight: 400 !important;
    line-height: 0.88 !important;
    letter-spacing: -0.045em !important;
}

.verdict-negative h3 {
    color: var(--ink);
}

.verdict h3 em {
    font-style: italic;
}

.verdict-body p {
    max-width: 480px;
    margin: 0;
    color: rgba(255,255,255,0.68);
    font-size: 0.78rem;
    line-height: 1.6;
}

.verdict-negative .verdict-body p {
    color: var(--muted);
}

.prob-wrap {
    padding: 0.15rem 0 1.25rem 0;
}

.prob-labels {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    color: rgba(255,255,255,0.55);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.prob-labels b {
    color: #ffffff;
    font-size: 0.88rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}

.verdict-negative .prob-labels {
    color: var(--muted);
}

.verdict-negative .prob-labels b {
    color: var(--ink);
}

.prob-track {
    position: relative;
    height: 9px;
    margin: 0.55rem 0 0.4rem 0;
    background: rgba(255,255,255,0.17);
}

.verdict-negative .prob-track {
    background: var(--soft);
}

.prob-fill {
    height: 100%;
    background: #ffffff;
}

.verdict-negative .prob-fill {
    background: var(--ink);
}

.prob-threshold {
    position: absolute;
    top: -4px;
    width: 1px;
    height: 17px;
    background: #ffffff;
    opacity: 0.9;
}

.verdict-negative .prob-threshold {
    background: var(--ink);
}

.prob-scale {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    color: rgba(255,255,255,0.42);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.06em;
}

.verdict-negative .prob-scale {
    color: var(--muted);
}

.verdict-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid rgba(255,255,255,0.22);
    margin: 0 -1.4rem;
}

.verdict-negative .verdict-metrics {
    border-top-color: var(--line);
}

.verdict-metrics > div {
    min-width: 0;
    padding: 0.85rem 1rem;
    border-right: 1px solid rgba(255,255,255,0.22);
}

.verdict-negative .verdict-metrics > div {
    border-right-color: var(--line);
}

.verdict-metrics > div:last-child {
    border-right: none;
}

.verdict-metrics span {
    display: block;
    margin-bottom: 0.28rem;
    color: rgba(255,255,255,0.48);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.verdict-negative .verdict-metrics span {
    color: var(--muted);
}

.verdict-metrics b {
    display: block;
    color: #ffffff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.verdict-negative .verdict-metrics b {
    color: var(--ink);
}

.fineprint {
    margin: 0.7rem 0 0 0;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    line-height: 1.55;
    letter-spacing: 0.01em;
}

.fig-label {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem;
    margin: 0.25rem 0 0.5rem 0;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.54rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.fig-label span:first-child {
    color: var(--ink);
}

.fig-note {
    margin: 0.65rem 0 0 0;
    color: var(--muted);
    font-size: 0.66rem;
    line-height: 1.55;
}

/* ── Model card ────────────────────────────────────────────────────────── */
.model-copy {
    color: var(--muted);
    font-size: 0.76rem;
    line-height: 1.65;
    margin: 0 0 1rem 0;
}

.model-copy b {
    color: var(--ink);
    font-weight: 500;
}

.pipeline {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.42rem;
    margin-top: 1rem;
}

.pipe-node {
    display: flex;
    min-height: 112px;
    flex-direction: column;
    justify-content: space-between;
    padding: 0.65rem 0.55rem;
    background: var(--white);
    border: 1px solid var(--line);
}

.pipe-node .num {
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.08em;
}

.pipe-node b {
    color: var(--ink);
    font-size: 0.72rem;
    font-weight: 600;
    line-height: 1.2;
}

.pipe-node small {
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.5rem;
    line-height: 1.3;
}

.pipe-node.active {
    background: var(--ink);
    border-color: var(--ink);
}

.pipe-node.active .num,
.pipe-node.active b,
.pipe-node.active small {
    color: #ffffff;
}

.pipe-node.active small {
    color: rgba(255,255,255,0.52);
}

.spec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-top: 1px solid var(--line);
    margin-top: 1.15rem;
}

.spec {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--line);
}

.spec:nth-child(odd) {
    padding-right: 1rem;
    border-right: 1px solid var(--line);
}

.spec:nth-child(even) {
    padding-left: 1rem;
}

.spec span {
    display: block;
    margin-bottom: 0.25rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.51rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.spec b {
    color: var(--ink);
    font-size: 0.76rem;
    font-weight: 500;
}

.perf-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    border-bottom: 1px solid var(--ink);
    padding-bottom: 0.65rem;
    margin-bottom: 0.15rem;
}

.perf-head b {
    color: var(--ink);
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.35rem;
    font-weight: 400;
    letter-spacing: -0.02em;
}

.perf-head span {
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.51rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.legend {
    display: flex;
    gap: 1rem;
    margin: 0.8rem 0 0.75rem 0;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.legend i {
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: 0.3rem;
    background: #c8c5bd;
    vertical-align: -0.05rem;
}

.legend i.improved {
    background: var(--ink);
}

.perf-row {
    margin: 0.85rem 0;
}

.perf-name {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 0.28rem;
}

.perf-name span {
    color: var(--ink);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.perf-name small {
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.5rem;
}

.perf-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.18rem 0;
}

.perf-tag {
    width: 31px;
    flex: 0 0 31px;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.47rem;
    letter-spacing: 0.03em;
}

.perf-track {
    flex: 1;
    height: 7px;
    background: var(--soft);
}

.perf-fill {
    height: 100%;
    background: var(--ink);
}

.perf-fill.base {
    background: #c8c5bd;
}

.perf-val {
    width: 46px;
    flex: 0 0 46px;
    text-align: right;
    color: var(--ink);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.57rem;
}

.cm-title {
    margin-top: 1.3rem;
    padding-top: 0.85rem;
    border-top: 1px solid var(--line);
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.51rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.cm {
    display: grid;
    grid-template-columns: 72px 1fr 1fr;
    border: 1px solid var(--line);
    margin-top: 0.45rem;
}

.cm div {
    min-height: 35px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-right: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    color: var(--ink);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
}

.cm div:nth-child(3n) {
    border-right: none;
}

.cm div:nth-last-child(-n+3) {
    border-bottom: none;
}

.cm .cm-head {
    background: var(--soft);
    color: var(--muted);
    font-size: 0.48rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.cm .cm-side {
    justify-content: flex-start;
    padding-left: 0.5rem;
    background: var(--soft);
    color: var(--muted);
    font-size: 0.48rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.cm .correct {
    background: var(--ink);
    color: #ffffff;
    font-size: 0.86rem;
}

.cm .wrong {
    color: var(--muted);
}

.data-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid var(--ink);
    border-bottom: 1px solid var(--line);
    margin-top: 1.55rem;
}

.data-cell {
    padding: 0.9rem 1rem;
    border-right: 1px solid var(--line);
}

.data-cell:last-child {
    border-right: none;
}

.data-cell span {
    display: block;
    margin-bottom: 0.28rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.51rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.data-cell b {
    color: var(--ink);
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.35rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    line-height: 1;
}

.notes {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    margin-top: 1.8rem;
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
}

.note {
    padding: 1rem 1.1rem 1.1rem 0;
    border-right: 1px solid var(--line);
}

.note:not(:first-child) {
    padding-left: 1.1rem;
}

.note:last-child {
    border-right: none;
}

.note span {
    display: block;
    margin-bottom: 0.4rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.51rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.note p {
    margin: 0;
    color: var(--ink);
    font-size: 0.72rem;
    line-height: 1.55;
}

/* ── Footer ───────────────────────────────────────────────────────────── */
.footer {
    margin-top: 3.3rem;
    padding: 1.6rem 1.6rem 1.7rem 1.6rem;
    background: var(--ink);
    color: #f3f1ea;
}

.footer-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1.5rem;
    border-bottom: 1px solid #2b2a27;
    padding-bottom: 0.9rem;
}

.footer-brand {
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.footer-brand .slash {
    color: #6e6b64;
    font-weight: 300;
}

.footer-meta {
    color: #7e7b74;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.52rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    text-align: right;
}

.footer-bottom {
    display: grid;
    grid-template-columns: 1.25fr 1fr 1fr;
    gap: 2rem;
    padding-top: 1rem;
}

.footer-bottom p {
    margin: 0;
    color: #8b8881;
    font-size: 0.66rem;
    line-height: 1.6;
}

.footer-bottom p b {
    color: #d8d5cd;
    font-weight: 500;
}

.footer-bottom a {
    color: #d8d5cd;
    text-decoration: underline;
    text-underline-offset: 3px;
}

/* ── Sidebar content ──────────────────────────────────────────────────── */
.side-brand {
    margin-bottom: 1.55rem;
    padding-bottom: 1.15rem;
    border-bottom: 1px solid #2b2a27;
}

.side-brand b {
    display: block;
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    line-height: 1;
    text-transform: uppercase;
    white-space: nowrap;
}

.side-brand b .slash {
    color: #9d9a92;
    font-weight: 300;
    margin: 0 0.08em;
}

.side-brand .side-brand-sub {
    display: block;
    margin-top: 0.38rem;
    color: #d6d2c9;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.52rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    line-height: 1;
    text-transform: uppercase;
}

.side-label {
    margin-bottom: 0.65rem;
    color: #6e6b64;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.51rem;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.side-title {
    margin: 0 0 0.6rem 0;
    color: #ffffff;
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.9rem;
    font-weight: 400;
    line-height: 0.95;
    letter-spacing: -0.025em;
}

.side-title em {
    font-style: italic;
}

.side-copy {
    margin: 0 0 1.2rem 0;
    color: #9d9a92 !important;
    font-size: 0.71rem;
    line-height: 1.6;
}

.side-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-top: 1px solid #2b2a27;
    margin-bottom: 1.2rem;
}

.side-metrics > div {
    padding: 0.75rem 0.5rem 0.75rem 0;
    border-bottom: 1px solid #2b2a27;
}

.side-metrics > div:nth-child(even) {
    padding-left: 0.75rem;
    border-left: 1px solid #2b2a27;
}

.side-metrics span {
    display: block;
    margin-bottom: 0.28rem;
    color: #6e6b64;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.48rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

.side-metrics b {
    color: #ffffff;
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.32rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    line-height: 1;
}

.side-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0.2rem 0 1.2rem 0;
}

.side-tags span {
    padding: 0.28rem 0.4rem;
    border: 1px solid #3b3935;
    color: #bdbab2;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.48rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.side-list {
    margin: 0;
    padding: 0;
    list-style: none;
}

.side-list li {
    position: relative;
    padding-left: 0.9rem;
    margin-bottom: 0.5rem;
    color: #9d9a92;
    font-size: 0.68rem;
    line-height: 1.5;
}

.side-list li::before {
    content: "—";
    position: absolute;
    left: 0;
    color: #ffffff;
}

.side-file {
    margin-top: 1.35rem;
    padding-top: 0.8rem;
    border-top: 1px solid #2b2a27;
    color: #5e5b55;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.47rem;
    line-height: 1.5;
    word-break: break-all;
}

.side-note {
    margin-top: 1.3rem;
    padding: 0.8rem;
    border: 1px solid #3b3935;
    color: #8b8881 !important;
    font-size: 0.62rem;
    line-height: 1.55;
}

@media (max-width: 1180px) {
    [data-testid="stAppViewBlockContainer"] {
        padding-left: 1.6rem;
        padding-right: 1.6rem;
    }
    .hero {
        gap: 2rem;
    }
    .pipeline {
        grid-template-columns: repeat(3, 1fr);
    }
}

@media (max-width: 900px) {
    .hero {
        grid-template-columns: 1fr;
    }
    .scan-card {
        min-height: 390px;
    }
    .section-head {
        align-items: flex-start;
        flex-direction: column;
        gap: 0.6rem;
    }
    .section-head > p {
        text-align: left;
        max-width: 520px;
    }
    .footer-bottom,
    .notes,
    .data-strip {
        grid-template-columns: 1fr 1fr;
    }
    .note {
        border-bottom: 1px solid var(--line);
    }
    .note:nth-child(2) {
        border-right: none;
    }
    .note:last-child {
        border-bottom: none;
    }
    .data-cell:nth-child(2) {
        border-right: none;
    }
    .data-cell:nth-child(-n+2) {
        border-bottom: 1px solid var(--line);
    }
}

@media (max-width: 720px) {
    .topnav {
        display: none;
    }
    .topbar {
        margin-bottom: 1.8rem;
    }
    .hero h1 {
        font-size: 3.5rem !important;
    }
    .verdict-metrics {
        grid-template-columns: 1fr 1fr;
    }
    .verdict-metrics > div:nth-child(2) {
        border-right: none;
    }
    .verdict-metrics > div:nth-child(-n+2) {
        border-bottom: 1px solid rgba(255,255,255,0.22);
    }
    .verdict-negative .verdict-metrics > div:nth-child(-n+2) {
        border-bottom-color: var(--line);
    }
    .pipeline {
        grid-template-columns: 1fr 1fr;
    }
    .footer-bottom,
    .notes,
    .data-strip {
        grid-template-columns: 1fr;
    }
    .footer-bottom {
        gap: 1rem;
    }
    .note,
    .note:not(:first-child) {
        padding-left: 0;
        border-right: none;
    }
    .data-cell {
        border-right: none;
    }
    .data-cell:not(:last-child) {
        border-bottom: 1px solid var(--line);
    }
}
</style>
"""


# ── HTML fragments ─────────────────────────────────────────────────────────
def topbar_html() -> str:
    return f"""
    <div class="topbar">
        <div class="brand">
            <span class="brand-mark">+</span>
            <span>PNEUMO<span class="slash">/</span>CXR</span>
        </div>
        <nav class="topnav">
            <a href="#diagnose">01 / Diagnose</a>
            <a href="#model">02 / Model card</a>
            <a href="#notes">03 / Notes</a>
        </nav>
        <div class="top-meta">
            <span class="status-dot"></span>
            Model 01 · Local inference
        </div>
    </div>
    """


def hero_html() -> str:
    scan_uri = image_data_uri(str(HERO_SCAN))
    heat_uri = image_data_uri(str(HERO_HEATMAP))
    scan_img = (
        f'<img class="scan" src="{scan_uri}" alt="Reference chest radiograph sample">'
        if scan_uri
        else ""
    )
    heat_img = (
        f'<img src="{heat_uri}" alt="Grad-CAM attention preview">'
        if heat_uri
        else ""
    )
    return f"""
    <section class="hero">
        <div class="hero-copy">
            <div class="eyebrow">AI-assisted chest radiography · Model 01</div>
            <h1>Pneumonia,<br><em>in plain sight.</em></h1>
            <p class="hero-lede">
                A monochrome screening workspace for frontal chest X-rays.
                MobileNetV2 with CBAM attention reads the scan, returns a probability,
                and shows you the regions that moved the decision.
            </p>
            <div class="hero-actions">
                <a class="btn-primary" href="#diagnose">Analyse a scan <span>↓</span></a>
                <span class="hero-note">Research and education only. Not a diagnostic device.</span>
            </div>
            <div class="hero-stats">
                <div><span>Test accuracy</span><b>{pct(IMPROVED['accuracy'])}</b></div>
                <div><span>Sensitivity</span><b>{pct(IMPROVED['recall'])}</b></div>
                <div><span>ROC–AUC</span><b>{IMPROVED['roc_auc']:.4f}</b></div>
            </div>
        </div>
        <div class="scan-card">
            {scan_img}
            <div class="scan-grid"></div>
            <span class="corner tl"></span><span class="corner tr"></span>
            <span class="corner bl"></span><span class="corner br"></span>
            <div class="crosshair-h"></div>
            <div class="crosshair-v"></div>
            <div class="scan-top">
                <span>Reference / 001</span>
                <span>PA view · demo</span>
            </div>
            <div class="scan-bottom">
                <div>
                    <div class="scan-pred-label">Model output · pneumonia</div>
                    <div class="scan-pred">Detected <b>0.96</b></div>
                </div>
                <div class="heat-inset">
                    {heat_img}
                    <span>CBAM attention</span>
                </div>
            </div>
        </div>
    </section>
    """


def section_head(section_id: str, number: str, title: str, description: str) -> str:
    return f"""
    <div class="section-head" id="{section_id}">
        <div>
            <div class="section-id">{number}</div>
            <h2>{title}</h2>
        </div>
        <p>{description}</p>
    </div>
    """


def empty_state_html() -> str:
    return """
    <div class="empty-state">
        <div class="empty-icon">+</div>
        <div class="empty-kicker">Awaiting input</div>
        <h3>No scan<br><em>loaded.</em></h3>
        <p>Choose a chest X-ray on the left, or open one of the two held-out test-set samples.
        The model will return a probability, a threshold-aware decision, and a Grad-CAM attention map.</p>
        <div class="empty-meta">Model 01 · MobileNetV2 + CBAM · 224 × 224</div>
    </div>
    """


def verdict_html(
    probability: float,
    threshold: float,
    source_label: str,
) -> str:
    positive = probability >= threshold
    band = risk_band(probability)
    normal_probability = 1.0 - probability
    if positive:
        state_class = "verdict-positive"
        headline = "Pneumonia<br><em>detected.</em>"
        description = (
            "The model found patterns consistent with pneumonia above the selected "
            "decision threshold. Review the attention map and seek clinical correlation."
        )
        kicker = "Primary finding · positive screen"
    else:
        state_class = "verdict-negative"
        headline = "No pneumonia<br><em>detected.</em>"
        description = (
            "The model did not find a pneumonia pattern above the selected decision "
            "threshold. This is a negative screen, not a ruled-out diagnosis."
        )
        kicker = "Primary finding · negative screen"

    return f"""
    <div class="verdict {state_class}">
        <div class="verdict-head">
            <span>Model output</span>
            <span>{escape(MODEL_LABEL)} · 0.5 reference</span>
        </div>
        <div class="verdict-body">
            <div class="verdict-kicker">{kicker}</div>
            <h3>{headline}</h3>
            <p>{description}</p>
        </div>
        <div class="prob-wrap">
            <div class="prob-labels">
                <span>Pneumonia probability</span>
                <b>{pct(probability)}</b>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{probability * 100:.1f}%"></div>
                <div class="prob-threshold" style="left:{threshold * 100:.1f}%"></div>
            </div>
            <div class="prob-scale">
                <span>0%</span>
                <span>Threshold {threshold * 100:.0f}%</span>
                <span>100%</span>
            </div>
        </div>
        <div class="verdict-metrics">
            <div><span>Normal</span><b>{pct(normal_probability)}</b></div>
            <div><span>Pneumonia</span><b>{pct(probability)}</b></div>
            <div><span>Band</span><b>{band}</b></div>
            <div><span>Source</span><b title="{escape(source_label)}">{escape(source_label)}</b></div>
        </div>
    </div>
    """


def model_card_html() -> str:
    metric_rows = []
    for label, key, note in [
        ("Accuracy", "accuracy", "overall correctness"),
        ("Precision", "precision", "positive predictive value"),
        ("Recall", "recall", "sensitivity"),
        ("F1 score", "f1_score", "precision / recall balance"),
        ("ROC–AUC", "roc_auc", "ranking quality"),
    ]:
        base_value = float(BASELINE[key])
        improved_value = float(IMPROVED[key])
        # Keep each row compact and unindented: Markdown turns indented HTML
        # that follows a blank line into a fenced code block.
        metric_rows.append(
            '<div class="perf-row">'
            f'<div class="perf-name"><span>{label}</span><small>{note}</small></div>'
            '<div class="perf-bar"><span class="perf-tag">BASE</span>'
            f'<div class="perf-track"><div class="perf-fill base" style="width:{bar_width(base_value):.1f}%"></div></div>'
            f'<span class="perf-val">{pct(base_value)}</span></div>'
            '<div class="perf-bar"><span class="perf-tag">IMPR</span>'
            f'<div class="perf-track"><div class="perf-fill" style="width:{bar_width(improved_value):.1f}%"></div></div>'
            f'<span class="perf-val">{pct(improved_value)}</span></div>'
            "</div>"
        )

    report = BASELINE.get("classification_report", {})
    normal_support = int(report.get("NORMAL", {}).get("support", 234))
    pneumonia_support = int(report.get("PNEUMONIA", {}).get("support", 390))
    tp = IMPROVED["recall"] * pneumonia_support
    fp = tp / IMPROVED["precision"] - tp if IMPROVED["precision"] else 0.0
    fn = pneumonia_support - tp
    tn = normal_support - fp
    tp, fp, fn, tn = (int(round(value)) for value in (tp, fp, fn, tn))

    return f"""
    <div class="model-copy">
        <b>MobileNetV2</b> is the feature extractor. <b>CBAM</b> refines its 7 × 7 × 1280
        feature map with channel and spatial attention before global pooling. A compact
        classifier head — dense 256, dropout, sigmoid — returns one pneumonia probability.
    </div>
    <div class="pipeline">
        <div class="pipe-node"><span class="num">01 / BACKBONE</span><b>MobileNetV2</b><small>2.2M pretrained</small></div>
        <div class="pipe-node active"><span class="num">02 / ATTENTION</span><b>CBAM</b><small>channel + spatial</small></div>
        <div class="pipe-node"><span class="num">03 / POOL</span><b>Global avg.</b><small>7 × 7 → 1280</small></div>
        <div class="pipe-node"><span class="num">04 / HEAD</span><b>Dense 256</b><small>ReLU + dropout</small></div>
        <div class="pipe-node"><span class="num">05 / OUTPUT</span><b>Sigmoid</b><small>P(pneumonia)</small></div>
    </div>
    <div class="spec-grid">
        <div class="spec"><span>Input</span><b>224 × 224 × 3</b></div>
        <div class="spec"><span>Task</span><b>Binary classification</b></div>
        <div class="spec"><span>Loss</span><b>Focal · γ 2.0 · α 0.25</b></div>
        <div class="spec"><span>Training</span><b>Two-phase transfer</b></div>
        <div class="spec"><span>Parameters</span><b>2.80M total</b></div>
        <div class="spec"><span>Output</span><b>One sigmoid score</b></div>
    </div>
    <div class="perf-head">
        <b>Test-set comparison</b>
        <span>624 images · 0.50 threshold</span>
    </div>
    <div class="legend">
        <span><i></i>Baseline</span>
        <span><i class="improved"></i>Improved · deployed</span>
    </div>
    {''.join(metric_rows)}
    <div class="cm-title">
        <span>Improved model · confusion matrix</span>
        <span>counts derived from reported metrics</span>
    </div>
    <div class="cm">
        <div class="cm-head"></div>
        <div class="cm-head">Pred. normal</div>
        <div class="cm-head">Pred. pneumonia</div>
        <div class="cm-side">Actual normal</div>
        <div class="correct">{tn}</div>
        <div class="wrong">{fp}</div>
        <div class="cm-side">Actual pneumonia</div>
        <div class="wrong">{fn}</div>
        <div class="correct">{tp}</div>
    </div>
    """


def data_strip_html() -> str:
    return """
    <div class="data-strip">
        <div class="data-cell"><span>Dataset images</span><b>5,856</b></div>
        <div class="data-cell"><span>Test set</span><b>624</b></div>
        <div class="data-cell"><span>Pneumonia share</span><b>73%</b></div>
        <div class="data-cell"><span>Model input</span><b>224²</b></div>
    </div>
    """


def notes_html() -> str:
    return """
    <div class="notes" id="notes">
        <div class="note">
            <span>Intended use</span>
            <p>Education, research, and demonstration of computer-aided chest X-ray
            screening. Always interpret alongside clinical history and a qualified reader.</p>
        </div>
        <div class="note">
            <span>Not intended for</span>
            <p>Primary diagnosis, triage, or ruling out disease. A negative screen does
            not exclude pneumonia, and a positive screen is not a diagnosis.</p>
        </div>
        <div class="note">
            <span>Known limitations</span>
            <p>Pediatric, single-center Kaggle cohort; binary labels only; no age, sex,
            or view metadata; validation split was very small. Performance may not transfer.</p>
        </div>
    </div>
    """


def footer_html() -> str:
    return """
    <div class="footer">
        <div class="footer-top">
            <div class="footer-brand">PNEUMO<span class="slash">/</span>CXR</div>
            <div class="footer-meta">AI Lab Project · 2026 · Model 01</div>
        </div>
        <div class="footer-bottom">
            <p><b>Built with</b> MobileNetV2, CBAM attention, focal loss, TensorFlow,
            Grad-CAM, and Streamlit.</p>
            <p><b>Data</b> Chest X-Ray Images (Pneumonia), Kermany et al. / Kaggle.
            Held-out test samples are used only for this demo.</p>
            <p><b>Notice</b> Not a medical device. For research and education only.
            Consult a qualified clinician for any medical decision.</p>
        </div>
    </div>
    """


def sidebar_html() -> str:
    return f"""
    <div class="side-brand">
        <div>
            <b>PNEUMO<span class="slash">/</span>CXR</b>
            <span class="side-brand-sub">Chest radiography AI</span>
        </div>
    </div>
    <div class="side-label">Model 01 / Deployed</div>
    <div class="side-title">MobileNetV2<br><em>+ CBAM.</em></div>
    <p class="side-copy">A binary pneumonia classifier trained on 5,856 pediatric chest
    X-rays. The attention module is what makes the model's focus visible.</p>
    <div class="side-metrics">
        <div><span>Accuracy</span><b>{pct(IMPROVED['accuracy'])}</b></div>
        <div><span>Recall</span><b>{pct(IMPROVED['recall'])}</b></div>
        <div><span>ROC–AUC</span><b>{IMPROVED['roc_auc']:.4f}</b></div>
        <div><span>F1 score</span><b>{pct(IMPROVED['f1_score'])}</b></div>
    </div>
    <div class="side-label">Techniques</div>
    <div class="side-tags">
        <span>Transfer learning</span>
        <span>CBAM</span>
        <span>Focal loss</span>
        <span>Grad-CAM</span>
    </div>
    <div class="side-label">At a glance</div>
    <ul class="side-list">
        <li>Input: 224 × 224 RGB chest radiograph</li>
        <li>Output: P(pneumonia), 0 to 1</li>
        <li>Decision threshold adjustable per scan</li>
        <li>Attention measured on the CBAM feature map</li>
    </ul>
    <div class="side-note">
        <b>Research use only.</b> This is not a medical device and does not provide a
        diagnosis. A negative result does not rule out pneumonia.
    </div>
    <div class="side-file">checkpoint / models/improved/mobilenet_improved.keras</div>
    """


# ── App ────────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    if "sample_key" not in st.session_state:
        st.session_state.sample_key = None
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = 0

    with st.sidebar:
        st.markdown(sidebar_html(), unsafe_allow_html=True)
        if REPORT_PDF.exists():
            st.download_button(
                "Project report · PDF",
                data=REPORT_PDF.read_bytes(),
                file_name="Pneumonia_Detection_Report.pdf",
                mime="application/pdf",
                width="stretch",
                key="sidebar_report_download",
            )
        st.caption("PNEUMO/CXR is a student AI Lab project. It is not affiliated with any clinical provider.")

    st.markdown(topbar_html(), unsafe_allow_html=True)
    st.markdown(hero_html(), unsafe_allow_html=True)
    st.markdown(
        section_head(
            "diagnose",
            "01 / Diagnose",
            "Read a scan.",
            "Upload a frontal chest radiograph, or open a held-out test sample. "
            "Every result includes the probability, the threshold-aware decision, and a Grad-CAM attention map.",
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.82, 1.18], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-label"><span>Input / chest X-ray</span><b>01</b></div>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Chest X-ray image",
                type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
                label_visibility="collapsed",
                key=f"cxr_upload_{st.session_state.uploader_nonce}",
            )
            st.caption("Frontal PA or AP view · JPG, PNG, BMP, TIFF · 25 MB max")

            st.markdown(
                '<div class="panel-label" style="margin-top:1.2rem;"><span>Or open a sample scan</span><b>02</b></div>',
                unsafe_allow_html=True,
            )
            sample_col_a, sample_col_b = st.columns(2, gap="small")
            with sample_col_a:
                if st.button(
                    SAMPLES["normal"]["short"],
                    type="primary" if st.session_state.sample_key == "normal" else "secondary",
                    width="stretch",
                    key="sample_normal_button",
                ):
                    st.session_state.sample_key = "normal"
                    st.session_state.uploader_nonce += 1
                    st.rerun()
            with sample_col_b:
                if st.button(
                    SAMPLES["pneumonia"]["short"],
                    type="primary" if st.session_state.sample_key == "pneumonia" else "secondary",
                    width="stretch",
                    key="sample_pneumonia_button",
                ):
                    st.session_state.sample_key = "pneumonia"
                    st.session_state.uploader_nonce += 1
                    st.rerun()

            if st.session_state.sample_key:
                sample = SAMPLES[st.session_state.sample_key]
                st.markdown(
                    f'<div class="sample-note">Active sample: {escape(sample["label"])} · '
                    f'ground truth {escape(sample["truth"])}. From the held-out test set.</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Clear sample",
                    type="tertiary",
                    width="stretch",
                    key="clear_sample_button",
                ):
                    st.session_state.sample_key = None
                    st.rerun()

            st.markdown(
                '<div class="guidance"><span>For best results</span>'
                "Use a full-lung-field frontal radiograph. The image is resized to "
                "224 × 224 and normalized to [0, 1]. No CLAHE or contrast enhancement is applied.</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="panel-label" style="margin-top:1.2rem;"><span>Decision threshold</span><b>03</b></div>',
                unsafe_allow_html=True,
            )
            threshold = st.slider(
                "Decision threshold",
                0.20,
                0.80,
                0.50,
                0.05,
                label_visibility="collapsed",
                help=(
                    "Lower thresholds increase sensitivity (fewer missed pneumonia cases) "
                    "and usually increase false positives."
                ),
            )
            st.caption(
                "Lower thresholds trade specificity for sensitivity. Reported test metrics use 0.50."
            )

    # Resolve the active input.
    image = None
    source_label = ""
    source_truth = None
    input_error = None

    if uploaded is not None:
        if getattr(uploaded, "size", 0) > 25 * 1024 * 1024:
            input_error = "This file is larger than 25 MB. Please upload a smaller image."
        else:
            st.session_state.sample_key = None
            try:
                image = open_image(uploaded.getvalue())
                source_label = uploaded.name
            except Exception:
                input_error = "The file could not be read as an image. Try a JPG, PNG, or BMP."
    elif st.session_state.sample_key:
        sample = SAMPLES[st.session_state.sample_key]
        try:
            image = open_image(sample["path"].read_bytes())
            source_label = sample["label"]
            source_truth = sample["truth"]
        except Exception:
            input_error = "The sample image could not be loaded. Check the data/samples folder."

    with right:
        if input_error:
            st.error(input_error)
            st.markdown(empty_state_html(), unsafe_allow_html=True)
        elif image is None:
            st.markdown(empty_state_html(), unsafe_allow_html=True)
        else:
            model_error = None
            try:
                with st.spinner("Loading MobileNetV2 + CBAM weights…"):
                    model, grad_model, target_layer = load_assets()
            except Exception as exc:
                model_error = str(exc)
                model, grad_model, target_layer = None, None, None

            if model_error:
                st.error(
                    "The model weights could not be loaded. Make sure "
                    "models/improved/mobilenet_improved.keras is present. "
                    f"Details: {model_error}"
                )
            else:
                with st.spinner("Reading the radiograph…"):
                    model_input, resized = preprocess_image(image)
                    probability = predict_probability(model, model_input)
                    positive = probability >= threshold
                    base_rgb, heat_rgb, overlay_rgb = make_gradcam(
                        grad_model, model_input, positive
                    )

                st.markdown(
                    verdict_html(probability, threshold, source_label),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p class="fineprint">The score is a model probability, not a '
                    "calibrated clinical risk. The threshold marker shows the active decision "
                    "cut-off; the displayed band is independent of the threshold.</p>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="panel-label" style="margin-top:1.25rem;">'
                    "<span>Visual evidence / Grad-CAM</span><b>CBAM</b></div>",
                    unsafe_allow_html=True,
                )
                fig_a, fig_b, fig_c = st.columns(3, gap="small")
                with fig_a:
                    st.markdown(
                        '<div class="fig-label"><span>01 / Input</span><span>224 × 224</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.image(base_rgb, width="stretch")
                with fig_b:
                    st.markdown(
                        '<div class="fig-label"><span>02 / Attention</span><span>white = focus</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.image(heat_rgb, width="stretch")
                with fig_c:
                    st.markdown(
                        '<div class="fig-label"><span>03 / Overlay</span><span>screen blend</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.image(overlay_rgb, width="stretch")
                st.markdown(
                    '<p class="fig-note">Grad-CAM is computed from the CBAM feature map. '
                    "Bright regions contributed more to the selected class score. It is a "
                    "coarse 7 × 7 attention map, not a lesion segmentation.</p>",
                    unsafe_allow_html=True,
                )

                result_payload = {
                    "app": "PNEUMO/CXR",
                    "model": MODEL_LABEL,
                    "checkpoint": "models/improved/mobilenet_improved.keras",
                    "attention_layer": target_layer,
                    "input_source": source_label,
                    "ground_truth": source_truth,
                    "pneumonia_probability": round(float(probability), 6),
                    "decision_threshold": round(float(threshold), 2),
                    "decision": "PNEUMONIA" if positive else "NORMAL",
                    "risk_band": risk_band(probability),
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "disclaimer": (
                        "Research and education only. Not a medical device. "
                        "A negative result does not rule out pneumonia."
                    ),
                }

                overlay_large = Image.fromarray(overlay_rgb).resize(
                    (448, 448), Image.Resampling.LANCZOS
                )
                download_a, download_b = st.columns(2, gap="small")
                with download_a:
                    st.download_button(
                        "Download result · JSON",
                        data=json.dumps(result_payload, indent=2),
                        file_name="pneumo_cxr_result.json",
                        mime="application/json",
                        width="stretch",
                        key="download_result_json",
                    )
                with download_b:
                    st.download_button(
                        "Download attention · PNG",
                        data=png_bytes(np.asarray(overlay_large)),
                        file_name="pneumo_cxr_attention.png",
                        mime="image/png",
                        width="stretch",
                        key="download_attention_png",
                    )

    st.markdown(
        section_head(
            "model",
            "02 / Model card",
            "Under the hood.",
            "A lightweight backbone, refined with attention, then fine-tuned end-to-end. "
            "The deployed checkpoint is the improved MobileNetV2 + CBAM model.",
        ),
        unsafe_allow_html=True,
    )

    model_left, model_right = st.columns([1.06, 0.94], gap="large")
    with model_left:
        st.markdown('<div class="panel-label"><span>Architecture</span><b>01</b></div>', unsafe_allow_html=True)
        st.markdown(model_card_html(), unsafe_allow_html=True)
    with model_right:
        st.markdown(
            '<div class="panel-label"><span>Test performance</span><b>02</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="model-copy">Bars use a shared 80–100% scale so that the small '
            "differences between the baseline and the improved model remain legible. "
            "The deployed model was selected for recall and F1, not ROC–AUC.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(data_strip_html(), unsafe_allow_html=True)
        st.markdown(
            '<div class="notes" style="grid-template-columns:1fr; margin-top:1.15rem;">'
            "<div class=\"note\" style=\"border-right:none;\">"
            "<span>Dataset</span><p>5,856 anterior-posterior pediatric chest radiographs "
            "from Guangzhou Women and Children's Medical Center, distributed via Kaggle. "
            "The validation split contains only 16 images, so validation estimates are noisy.</p></div>"
            "<div class=\"note\" style=\"border-right:none; border-bottom:none;\">"
            "<span>Training</span><p>Two-phase transfer learning: frozen backbone, then "
            "fine-tuned from layer 100. Focal loss (γ = 2.0, α = 0.25) and augmentation "
            "address the 73 / 27 class imbalance.</p></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(notes_html(), unsafe_allow_html=True)
    st.markdown(footer_html(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
