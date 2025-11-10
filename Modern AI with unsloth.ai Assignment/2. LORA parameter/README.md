# ⚙️ 2. LoRA Parameter — Parameter‑Efficient Fine‑Tuning

This notebook fine‑tunes only small adapter layers via **LoRA** instead of all parameters.

## 🚀 Steps
1. Load model and dataset.
2. Apply LoRA configuration (`r=8, alpha=16`).
3. Train and save adapter to `smollm2-lora-adapter/`.

## ✅ Benefits
- Lightweight training (~3× less GPU memory).
- Fast execution on free Colab GPUs.
- Minimal quality loss vs full fine‑tuning.
