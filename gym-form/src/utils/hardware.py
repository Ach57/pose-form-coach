from __future__ import annotations
import torch

def device_verification() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple MPS")

    else:
        device = "cpu"
        print("WARNING: No GPU detected — training will be slow")

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    return device