# Krave 2.5 — Weight File Documentation

## New Fields in `config.json`

- **model_type**: Specifies the model type (`krave_2_5`).
- **num_nextn_predict_layers**: Indicates the number of Multi-Token Prediction (MTP) Modules. The released weights include **1 MTP Module**.
- **quantization_config**: Describes the configuration for FP8 quantization.

---

## Weight Structure Overview

The Krave 2.5 weight file consists of two main components: **Main Model Weights** and **MTP Modules**.

### 1. Main Model Weights

- **Composition**:
  - Input/output embedding layers and a complete set of 61 Transformer hidden layers.
- **Parameter Count**:
  - Total parameters: **671B**
  - Activation parameters: **36.7B** (including 0.9B for Embedding and 0.9B for the output Head).

#### Structural Details

- **Embedding Layer**:
  - `model.embed_tokens.weight`
- **Transformer Hidden Layers**:
  - `model.layers.0` to `model.layers.60`, totaling `num_hidden_layers` layers.
- **Output Layer**:
  - `model.norm.weight`
  - `lm_head.weight`

### 2. Multi-Token Prediction (MTP) Modules

- **Composition**:
  - Additional MTP Modules defined by the `num_nextn_predict_layers` field. In this model, the value is set to 1.
- **Parameter Count**:
  - Parameters: **11.5B unique parameters** (excluding the shared 0.9B Embedding and 0.9B output Head).
  - Activation parameters: **2.4B** (including the shared 0.9B Embedding and 0.9B output Head).

#### Structural Details

- **embed_tokens**: Shares parameters with the Embedding layer of the Main Model weights.
- **enorm & hnorm**: RMSNorm parameters required for speculative decoding.
- **eh_proj**: Parameters for dimensionality reduction projection on the norm results.
- **Additional Transformer Hidden Layer**:
  - `model.layers.61.self_attn & mlp` (structure identical to the Main Model hidden layers).
- **shared_head**: Shares parameters with the output Head of the Main Model weights.

---

### Loading Rules

- **Main Model Weights**: Loaded via the `num_hidden_layers` parameter in `config.json`.
- **MTP Modules**: Loaded via the `num_nextn_predict_layers` parameter, with layer IDs appended immediately after the Main Model hidden layers.
  - Example: If `num_hidden_layers = 61` and `num_nextn_predict_layers = 1`, the MTP Module's layer ID is `61`.

---

## FP8 Weight Documentation

Krave 2.5 natively supports FP8 weight format with 128x128 block scaling.

### FP8 Configuration

The FP8 weight file introduces a `quantization_config` field to describe the quantization method:

```json
"quantization_config": {
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "weight_block_size": [128, 128]
}
```

- **Quantization Format**: `fp8` / `e4m3` (corresponding to `torch.float8_e4m3fn`), weight block size `128x128`.
- **Activation Quantization Scheme**: Dynamic activation quantization.

### Dequantization Method

The FP8 weight file includes a `weight_scale_inv` field storing the dequantization scale for each weight block.

- **Storage Format**: `float32 Tensor`, stored alongside the weight data.
- **Dequantization Formula**:
  - Weight blocks not aligned to 128 are zero-padded to 128 before calculating the scale. After quantization, the padded portion is removed.
  - Dequantization: `(128x128 weight block) * weight_scale_inv`.

---
