# LLaVA-1.5 VCD + ASD Reproduction

This repository contains my attempt to reproduce and combine **Visual Contrastive Decoding (VCD)** and **Activation Steering Decoding (ASD)** on **LLaVA-1.5** to reduce hallucinations in large vision–language models. The project was run under realistic constraints (Colab Pro, limited GPU, and small MSCOCO subsets), so it focuses as much on **engineering + evaluation pitfalls** as on raw performance gains.

> TL;DR: VCD clearly changes behavior under sampling-based decoding, ASD is hard to see at small scale, and subtle ~3% gains reported in the papers are basically invisible on 15–30 image subsets.

---

## 1. Project Goals

- Implement **VCD** on top of LLaVA-1.5’s decoding pipeline.
- Implement **ASD-style steering** via forward hooks on LLaVA-1.5’s hidden activations.
- Run a **POPE-style yes/no evaluation** on MSCOCO subsets to compare:
  - Baseline (vanilla LLaVA-1.5)
  - VCD only
  - VCD + ASD
- Document **why** the original reported accuracy improvements (≈2–3%) are hard to reproduce in a constrained setting.

---

## 2. Methods (High-Level)

### 2.1 Visual Contrastive Decoding (VCD)

- Idea: reduce hallucinations by changing the **decoding process** rather than retraining.
- For each step, the model generates logits using:
  - a **clean** image view
  - a **noisy/perturbed** image view  
- Tokens that are **overconfident in both views** get penalized, under the assumption that truly image-grounded tokens will be more stable across views.
- In this repo:
  - VCD is integrated near the `generate()` / decoding loop.
  - It modifies token logits before sampling, using the contrast between clean vs. perturbed image representations.

### 2.2 Activation Steering Decoding (ASD)

- Idea: intervene directly in the **hidden activations** of the model rather than just the logits.
- The ASD paper computes **steering vectors** by running PCA on differences between:
  - hallucinated outputs vs.
  - grounded / non-hallucinated outputs.
- These principal directions are then added/subtracted at inference time to “nudge” the model toward more faithful generations.
- In this repo:
  - I estimate steering vectors on a **small synthetic dataset** (only a few dozen examples).
  - I apply them with **small scaling factors (α, β)** via forward hooks on:
    - the vision–language projection layer
    - the final text transformer layer

### 2.3 LLaVA-1.5 Integration Details

Key implementation details I had to get right:

- **Environment**
  - Python 3.9 via **micromamba**
  - CUDA-matched **PyTorch** installed first
  - Pinned **Transformers** version so the custom LLaVA model class would register correctly
  - `device_map="auto"` and `low_cpu_mem_usage=True` for Colab Pro GPU limits

- **Imports / paths**
  - Added the VCD repo directories to `sys.path`:
    - `/content/VCD`
    - `/content/VCD/experiments`
  - Fixed `ModuleNotFoundError: 'llava'` by making sure the repo’s custom code was importable.

- **Vision tower alignment**
  - Explicitly initialized the **CLIP vision encoder** and ensured:
    - same `dtype` as the language model
    - same device
  - This avoided silent dtype/device mismatches that can mask subtle decoding effects.

- **Image token injection**
  - Simply writing `<image>` in the prompt does **nothing** unless the special image token is correctly inserted.
  - Used `tokenizer_image_token()` so `<image>` is mapped to the true image token index that activates the vision encoder.
  - Before this fix, LLaVA behaved like a **text-only LLaMA**, generating nearly identical captions for different images.

---

## 3. Evaluation Setup

### 3.1 Dataset

- **MSCOCO validation** subset.
- Due to Colab Pro constraints:
  - Started with **15 images**
  - Later expanded to **30 images**
- For each image, I generated **yes/no questions** about object presence (POPE-style), balancing positive and negative examples.

### 3.2 Modes Compared

1. **Baseline**: vanilla LLaVA-1.5
2. **VCD-only**
3. **VCD + ASD**

All modes used the *same* prompts and image set for a fair comparison.

### 3.3 Decoding Settings

I explicitly experimented with **two decoding regimes**:

1. **Sampling-based decoding**
   - `do_sample = True`
   - `temperature = 0.7`
   - `top_p = 0.9`

2. **Greedy decoding (deterministic)**
   - `do_sample = False`
   - `temperature = 0.0` (effectively ignored)
   - Used for the POPE-style yes/no evaluation.

---

## 4. Key Findings & Why Reproduction Failed

### 4.1 Accuracy Gaps Are Too Small for Tiny Sample Sizes

- Reported improvements in the papers:
  - VCD / ASD show **≈2–3%** gains over baselines on **thousands of COCO samples**.
- In my setup:
  - With 15 images, accuracies were very high and similar across modes.
  - Expanding from **15 → 30 images**:
    - Overall accuracy **dropped slightly** (e.g., ~95% → ~92%),
    - But **all three modes dropped together**.
- Result:  
  - No clear separation between baseline, VCD, and VCD + ASD.
  - At this scale, the variance / noise is larger than the expected effect size.

### 4.2 VCD Only Shows Up Under Sampling

- When using **sampling-based decoding** (`do_sample=True`):
  - VCD often produced **more cautious / less hallucinatory** captions compared to baseline.
  - Outputs diverged more often, indicating VCD was indeed influencing the logits.
- When switching to **greedy decoding**:
  - VCD and baseline outputs became **identical** most of the time.
- Likely explanation:
  - Under greedy decoding, the model always picks the **single highest-probability token**.
  - VCD’s logit adjustments are relatively small, often not enough to change the top-1 token.
  - Sampling-based decoding, however, is more sensitive to subtle probability shifts, so VCD can change generation diversity and specificity.

### 4.3 ASD Effect Was Essentially Invisible

- VCD + ASD outputs were **almost identical** to VCD alone under both decoding settings.
- Suspected reasons:
  - **Tiny steering dataset**: steering vectors computed on only a few dozen synthetic examples.
  - **Noisy PCA directions**: limited and noisy hallucinated vs. non-hallucinated pairs.
  - **Conservative scaling (small α, β)** to avoid breaking the model.
- Combined effect:  
  - ASD barely changed hidden activations, so VCD + ASD behaved like pure VCD.

### 4.4 POPE-style Evaluation Is Sensitive to Implementation Details

- Early versions of my POPE-style evaluator:
  - Used **overlapping token sets** for “Yes” and “No,” which collapsed the margin between them.
  - This made all modes look equally good.
- After fixing the token mapping:
  - Accuracies remained **very close** across modes.
  - With such a tiny dataset, even a real 3% improvement would be hard to distinguish from noise.