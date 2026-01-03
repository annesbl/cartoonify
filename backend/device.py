# backend/device.py
import torch


def get_device(prefer_mps: bool = False) -> torch.device:
    """
    prefer_mps=False  -> erzwingt CPU (dein Wunsch)
    prefer_mps=True   -> nutzt MPS falls verfügbar, sonst CPU
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    # CPU/MPS: float32 ist am stabilsten (float16 auf CPU ist oft langsam/fehleranfällig)
    return torch.float32
