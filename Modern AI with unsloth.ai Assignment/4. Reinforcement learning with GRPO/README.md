# 🚀 4. Reinforcement Learning with GRPO‑Lite (Self‑Play)

Introduces **GRPO‑Lite**, a lightweight self‑play reasoning method built on DPO.

## 🧩 Method
- Model generates multiple responses per prompt.
- Reward function scores reasoning and structure.
- Automatically creates “chosen” vs “rejected” pairs.

## ⚙️ Pipeline
1. Generate responses.
2. Score with heuristic reward.
3. Train with LoRA adapters using DPO.
4. Save results as `smollm2-135m-grpo-lite-lora/`.

## 🎯 Outcome
- Adds reasoning quality to model responses.
- 100% compatible with Colab free GPUs.
