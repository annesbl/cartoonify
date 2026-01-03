# backend/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class PassConfig:
    strength: float
    guidance_scale: float
    steps: int
    prompt_suffix: str
    negative_prompt: str


@dataclass(frozen=True)
class SimpsonifySettings:
    # Models
    base_model: str
    lora_path: str
    device: str

    # Global defaults
    default_prompt: str
    default_negative_prompt: str
    seed: Optional[int]
    lora_scale: float

    # Two-pass behavior
    two_pass_enabled: bool
    pass1: PassConfig
    pass2: PassConfig

    # Debug
    debug: bool


def pick_device(explicit: Optional[str] = None) -> str:
    """
    CPU-first für MacBook (ohne NVIDIA CUDA).
    - Wenn SD_DEVICE gesetzt ist:
        - "cpu" erzwingt CPU
        - "mps" nutzt Apple Metal (wenn verfügbar), sonst CPU
        - "cuda" wird ignoriert und fällt auf CPU zurück (damit es nicht crasht)
    - Wenn SD_DEVICE NICHT gesetzt ist: CPU (oder MPS wenn du willst)
    """
    exp = (explicit or "").strip().lower()

    # explizite Wahl
    if exp in {"cpu", "mps"}:
        if exp == "mps":
            try:
                import torch

                if torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
        return "cpu"

    # "cuda" oder irgendwas anderes: auf Mac sicher auf CPU fallen
    return "cpu"


def load_settings() -> SimpsonifySettings:
    # Base defaults (one place!)
    default_prompt = os.getenv(
        "SD_DEFAULT_PROMPT",
        "simpsons character portrait, thick black outline, clean lineart, flat colors, "
        "cel shading, yellow skin, 2D cartoon, simple background",
    )
    default_negative = os.getenv(
        "SD_DEFAULT_NEGATIVE",
        "photo, realistic, airbrush, smooth skin, beauty, makeup, cinematic lighting, "
        "depth of field, bokeh, detailed skin, pores, hdr, 3d render, blurry",
    )

    # Pass 1/2 defaults
    pass1_neg = os.getenv(
        "SD_PASS1_NEG",
        "photo, realistic, natural skin, pores, cinematic lighting, depth of field, bokeh, hdr, 3d render",
    )
    pass2_neg = os.getenv(
        "SD_PASS2_NEG", "photo, realistic, watercolor, painting, soft shading, blur"
    )

    return SimpsonifySettings(
        base_model=os.getenv("SD_BASE_MODEL", "runwayml/stable-diffusion-v1-5"),
        lora_path=os.getenv(
            "SD_LORA_PATH",
            "/root/simpsonify/simpsonify/backend/models/simpsons_style_lora-000008.safetensors",
        ),
        device=pick_device(os.getenv("SD_DEVICE")),
        default_prompt=default_prompt,
        default_negative_prompt=default_negative,
        seed=int(os.getenv("SD_SEED")) if os.getenv("SD_SEED") else None,
        lora_scale=float(os.getenv("SD_LORA_SCALE", "1.6")),
        two_pass_enabled=os.getenv("SD_TWO_PASS", "1") == "1",
        pass1=PassConfig(
            strength=float(os.getenv("SD_PASS1_STRENGTH", "0.55")),
            guidance_scale=float(os.getenv("SD_PASS1_GUIDANCE", "4.5")),
            steps=int(os.getenv("SD_PASS1_STEPS", "20")),
            prompt_suffix=os.getenv(
                "SD_PASS1_SUFFIX",
                "bright yellow skin, flat solid colors, no shading, simple shapes, 2D cel animation",
            ),
            negative_prompt=pass1_neg,
        ),
        pass2=PassConfig(
            strength=float(os.getenv("SD_PASS2_STRENGTH", "0.50")),
            guidance_scale=float(os.getenv("SD_PASS2_GUIDANCE", "6.0")),
            steps=int(os.getenv("SD_PASS2_STEPS", "40")),
            prompt_suffix=os.getenv(
                "SD_PASS2_SUFFIX",
                "thick black outline, clean lineart, flat colors, simple shapes, 2D cartoon TV show",
            ),
            negative_prompt=pass2_neg,
        ),
        debug=os.getenv("SD_DEBUG", "0") == "1",
    )


def settings_as_dict(s: SimpsonifySettings) -> Dict[str, Any]:
    return {
        "base_model": s.base_model,
        "lora_path": s.lora_path,
        "device": s.device,
        "default_prompt": s.default_prompt,
        "default_negative_prompt": s.default_negative_prompt,
        "seed": s.seed,
        "lora_scale": s.lora_scale,
        "two_pass_enabled": s.two_pass_enabled,
        "pass1": {
            "strength": s.pass1.strength,
            "guidance_scale": s.pass1.guidance_scale,
            "steps": s.pass1.steps,
            "prompt_suffix": s.pass1.prompt_suffix,
            "negative_prompt": s.pass1.negative_prompt,
        },
        "pass2": {
            "strength": s.pass2.strength,
            "guidance_scale": s.pass2.guidance_scale,
            "steps": s.pass2.steps,
            "prompt_suffix": s.pass2.prompt_suffix,
            "negative_prompt": s.pass2.negative_prompt,
        },
        "debug": s.debug,
    }
