# Krave 2.5

**Krave 2.5** is an open-source Mixture-of-Experts (MoE) language model for users who want to run their own local or private LLM setup.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Model Summary](#2-model-summary)
3. [Model Downloads](#3-model-downloads)
4. [How to Run Locally](#4-how-to-run-locally)
5. [Krave Engine](#5-krave-engine)
6. [License](#6-license)

---

## 1. Introduction

Krave 2.5 is a strong Mixture-of-Experts (MoE) language model with 671B total parameters and 37B activated per token. It adopts Multi-head Latent Attention (MLA) and a MoE architecture for efficient inference and cost-effective training. This repository is intended as an open-source LLM project, and users should provide their own weight files for it to function.

---

## 2. Model Summary

**Architecture**

- **Auxiliary-loss-free load balancing** — minimizes performance degradation while encouraging balanced load across experts.
- **Multi-Token Prediction (MTP)** — improves model performance and enables speculative decoding for faster inference.
- **Multi-head Latent Attention (MLA)** — efficient attention mechanism with low-rank key-value compression.

---

## 3. Model Downloads

This project does not ship model weights. To run Krave 2.5, you must provide your own compatible weight files.

---

## 4. How to Run Locally

### System Requirements

> Linux with Python 3.10+. Mac and Windows are not supported.

**Dependencies:**
```
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```

### Setup

Clone the repository:

```shell
git clone https://github.com/kraveorg/Krave-2.5.git
cd Krave-2.5
```

Install dependencies:

```shell
cd inference
pip install -r requirements.txt
```

Provide your own weight files in the expected checkpoint directory before running inference.

### Convert Weights

If you already have compatible weights, convert them to the required format:

```shell
python convert.py --hf-ckpt-path /path/to/your-weights \
  --save-path /path/to/Krave-2.5-Demo \
  --n-experts 256 --model-parallel 16
```

### Run — Interactive Mode

```shell
torchrun --nproc-per-node=8 inference/generate.py \
  --ckpt-path /path/to/your-weights \
  --config inference/configs/config_671B.json \
  --interactive --temperature 0.7 --max-new-tokens 200
```

### Run — Batch Mode

```shell
torchrun --nproc-per-node=8 inference/generate.py \
  --ckpt-path /path/to/your-weights \
  --config inference/configs/config_671B.json \
  --input-file prompts.txt
```

### Hardware Recommendations

- Minimum: 8x A100 (80GB) GPUs
- Recommended: 8x H100 (80GB) GPUs per node, 2 nodes
- NCCL for multi-GPU / multi-node communication

### Supported Inference Frameworks

- **SGLang** — BF16 and FP8, multi-node tensor parallelism
- **LMDeploy** — FP8 and BF16, offline and online serving
- **TensorRT-LLM** — BF16, INT4/INT8
- **vLLM** — FP8 and BF16, tensor and pipeline parallelism
- **LightLLM** — FP8 and BF16, single and multi-node
- **AMD GPU** — via SGLang, BF16 and FP8
- **Huawei Ascend NPU** — INT8 and BF16

---

## 5. Krave Engine

Krave 2.5 includes a built-in **Krave Engine** — a lightweight Python interface for loading and running the model programmatically.

```python
from engine import KraveEngine

engine = KraveEngine(
    ckpt_path="/path/to/your-weights",
    config="inference/configs/config_671B.json"
)

response = engine.generate("Explain quantum computing in simple terms.")
print(response)
```

See [`engine.py`](./engine.py) for full API documentation.

---

## 6. License

This code repository is licensed under the [MIT License](LICENSE-CODE). Model weights are subject to [the Model License](LICENSE-MODEL). Krave 2.5 supports commercial use.
