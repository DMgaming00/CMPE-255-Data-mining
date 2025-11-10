# 🧩 5. Continued Pretraining — Merge, Inference, Evaluation

Final notebook for merging and evaluating LoRA adapters in FP32 mode.

## ⚙️ Steps
1. Mount Google Drive (load adapters saved in Colab 4).
2. Auto‑detect LoRA folders (local or Drive).
3. Load model, attach adapter, and run inference.
4. (Optional) Merge LoRA and export zipped model.

## ✅ Features
- FP32 precision for error‑free execution.
- Offline‑safe (`local_files_only=True`).
- Simple evaluation using keyword heuristics.
