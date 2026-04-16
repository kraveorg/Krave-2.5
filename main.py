#!/usr/bin/env python3
"""
Krave 2.5 - Open Source Large Language Model
"""

import sys
import os

def print_banner():
    print("=" * 70)
    print("  Krave 2.5: Open Source Large Language Model")
    print("=" * 70)
    print()
    print("  A 671B parameter Mixture-of-Experts (MoE) language model.")
    print("  37B parameters are activated per token during inference.")
    print()
    print("  Key Features:")
    print("    - Multi-head Latent Attention (MLA)")
    print("    - Auxiliary-loss-free load balancing")
    print("    - Multi-Token Prediction (MTP) training objective")
    print("    - FP8 mixed precision inference support")
    print("    - Context length up to 128K tokens")
    print()
    print("=" * 70)
    print()

def print_hardware_requirements():
    print("  Hardware Requirements:")
    print("    - Multiple high-end NVIDIA or AMD GPUs")
    print("    - Recommended: 8x H100 (80GB) or equivalent")
    print("    - PyTorch 2.4.1 + Triton 3.0.0 (GPU required)")
    print("    - NCCL for multi-GPU communication")
    print()

def print_usage():
    print("  Usage (multi-GPU, from the inference/ directory):")
    print()
    print("    # Interactive mode")
    print("    torchrun --nproc-per-node=8 generate.py \\")
    print("        --ckpt-path /path/to/Krave-2.5 \\")
    print("        --config configs/config_671B.json \\")
    print("        --interactive")
    print()
    print("    # Batch mode")
    print("    torchrun --nproc-per-node=8 generate.py \\")
    print("        --ckpt-path /path/to/Krave-2.5 \\")
    print("        --config configs/config_671B.json \\")
    print("        --input-file prompts.txt \\")
    print("        --max-new-tokens 200")
    print()

def print_model_configs():
    import json
    config_dir = os.path.join(os.path.dirname(__file__), "inference", "configs")
    if os.path.isdir(config_dir):
        print("  Available Model Configurations:")
        for fname in sorted(os.listdir(config_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(config_dir, fname)
                try:
                    with open(fpath) as f:
                        cfg = json.load(f)
                    n_layers = cfg.get("n_layers", "?")
                    dim = cfg.get("dim", "?")
                    print(f"    - {fname}: {n_layers} layers, dim={dim}")
                except Exception:
                    print(f"    - {fname}")
        print()

def print_links():
    print("  Resources:")
    print("    - GitHub:  https://github.com/your-org/krave-2.5")
    print("    - Weights: https://huggingface.co/deepseek-ai/DeepSeek-V3 (base weights)")
    print()
    print("  Compatible Inference Frameworks:")
    print("    - SGLang, LMDeploy, vLLM, TensorRT-LLM, LightLLM")
    print()

def check_dependencies():
    print("  Checking installed dependencies...")
    deps = {
        "transformers": "4.46.3",
        "safetensors": "0.4.5",
    }
    for pkg, expected in deps.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            status = "OK" if version == expected else f"OK (v{version})"
            print(f"    [{status}] {pkg}")
        except ImportError:
            print(f"    [MISSING] {pkg} (required: {expected})")

    for pkg in ["torch", "triton"]:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"    [OK] {pkg} v{version}")
        except ImportError:
            print(f"    [NOT INSTALLED] {pkg} — requires GPU environment")

    print()

def main():
    print_banner()
    print_hardware_requirements()
    check_dependencies()
    print_model_configs()
    print_usage()
    print_links()
    print("  Krave 2.5 is open source and free to use.")
    print("  To run inference, deploy on a multi-GPU machine with the weights.")
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
