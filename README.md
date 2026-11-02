# PNEUMO/CXR

A black-and-white Streamlit workspace for pneumonia detection from frontal chest X-rays.
The app loads the trained **MobileNetV2 + CBAM + Focal Loss** checkpoint, returns a
pneumonia probability, applies an adjustable decision threshold, and renders a monochrome
Grad-CAM attention map.

![PNEUMO/CXR app preview](docs/app-preview.png)

## What it does

- Upload a chest X-ray (JPG, PNG, BMP, WEBP, TIFF) or open one of the held-out test samples.
- Get a model probability, a threshold-aware decision, and a risk band.
- Adjust the decision threshold from 0.20 to 0.80 to trade sensitivity for specificity.
- Inspect three views: normalized input, Grad-CAM attention, and a screen-blended overlay.
- Download the result as JSON or the attention overlay as PNG.
- Review the model card, baseline comparison, and the project PDF from the sidebar.

## Model

| Item | Value |
| --- | --- |
| Backbone | MobileNetV2 (ImageNet pretrained) |
| Attention | CBAM — channel + spatial |
| Loss | Focal loss, γ = 2.0, α = 0.25 |
| Input | 224 × 224 × 3, normalized to [0, 1] |
| Training | Two-phase transfer learning |
| Deployed checkpoint | `models/improved/mobilenet_improved.keras` |

### Test-set results

| Metric | Baseline | Improved (deployed) |
| --- | ---: | ---: |
| Accuracy | 87.98% | **88.14%** |
| Precision | 87.23% | 87.09% |
| Recall / sensitivity | 94.62% | **95.13%** |
| F1 score | 90.77% | **90.93%** |
| ROC–AUC | **95.82%** | 94.30% |

The improved model is deployed for its higher recall and F1. Reported metrics use a
0.50 decision threshold on the 624-image test split. The baseline has a higher ROC–AUC.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The weights are already in the repository at
`models/improved/mobilenet_improved.keras`, so no download step is required at runtime.
TensorFlow is CPU-only on native Windows for versions ≥ 2.11; inference on a single
224 × 224 image is still fast.

## Project structure

```text
Project/
├─ app.py                         # Streamlit application
├─ requirements.txt
├─ Pneumonia_Detection_Report.pdf
├─ generate_report.py
├─ .streamlit/
│  └─ config.toml                 # Monochrome Streamlit theme
├─ data/
│  └─ samples/                    # Held-out demo scans used by the app
├─ models/
│  ├─ baseline_results.json
│  ├─ mobilenet_pneumonia.h5      # Baseline checkpoint
│  └─ improved/
│     ├─ mobilenet_improved.keras # Deployed checkpoint
│     ├─ best_improved_model.keras
│     └─ improved_results.json
└─ notebooks/
   ├─ pneumonia_train.ipynb
   └─ pneumonia_improved.ipynb
```

## Notes and limitations

- The dataset contains 5,856 pediatric, single-center chest X-rays from Guangzhou;
  the model may not generalize to adults or other institutions.
- The task is binary (NORMAL vs PNEUMONIA). It does not distinguish bacterial from
  viral pneumonia or localize lesions.
- The validation split contains only 16 images, so validation estimates are noisy.
- Grad-CAM is a coarse 7 × 7 attention map computed from the CBAM feature map. It is
  not a lesion segmentation or a causal explanation.
- Do not use this app for clinical diagnosis, triage, or ruling out disease.

## Dataset and report

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia),
  Kermany et al.
- Full write-up: [`Pneumonia_Detection_Report.pdf`](Pneumonia_Detection_Report.pdf)

## Disclaimer

This is a student AI Lab project for research and education only. It is not a medical
device. Always consult a qualified clinician for any medical decision.
