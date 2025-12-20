# backend/services/sd_service.py
"""
Stable Diffusion img2img + LoRA service (Simpsonify – final tuned).

- Lazy-load pipeline
- CUDA fp16 / CPU-MPS fp32
- Safety checker disabled (local dev)
- PEFT LoRA with adapter control
- Deterministic seeding
- Two-pass img2img:
  Pass 1: force yellow + flat colors
  Pass 2: enforce outlines + Simpsons style
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file as safetensors_load_file

from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import is_peft_available


# ============================
# Config
# ============================

@dataclass
class SDConfig:
    base_model: str
    lora_path: str
    device: str
    guidance_scale: float
    strength: float
    num_inference_steps: int
    seed: Optional[int]
    lora_scale: float
    debug: bool


_PIPE: Optional[StableDiffusionImg2ImgPipeline] = None
_CFG: Optional[SDConfig] = None
_DID_AB = False

_ADAPTER_NAME = "simpsons"


def _pick_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config_from_env() -> SDConfig:
    return SDConfig(
        base_model=os.getenv("SD_BASE_MODEL", "runwayml/stable-diffusion-v1-5"),
        lora_path=os.getenv(
            "SD_LORA_PATH",
            "/root/simpsonify/simpsonify/backend/models/simpsons_style_lora-000008.safetensors",
        ),
        device=_pick_device(os.getenv("SD_DEVICE")),
        guidance_scale=float(os.getenv("SD_GUIDANCE", "7.8")),
        strength=float(os.getenv("SD_STRENGTH", "0.80")),
        num_inference_steps=int(os.getenv("SD_STEPS", "40")),
        seed=int(os.getenv("SD_SEED")) if os.getenv("SD_SEED") else None,
        lora_scale=float(os.getenv("SD_LORA_SCALE", "2.0")),
        debug=os.getenv("SD_DEBUG", "0") == "1",
    )


# ============================
# Pipeline
# ============================

def _load_pipeline(cfg: SDConfig) -> StableDiffusionImg2ImgPipeline:
    dtype = torch.float16 if cfg.device == "cuda" else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    ).to(cfg.device)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    lora_path = str(Path(cfg.lora_path).resolve())

    pipe.load_lora_weights(lora_path, adapter_name=_ADAPTER_NAME)
    pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[cfg.lora_scale])

    if cfg.debug:
        print("\n=== PIPE INIT ===")
        print("DEVICE:", cfg.device)
        print("DTYPE:", dtype)
        print("LORA:", lora_path)
        print("PEFT:", is_peft_available())

        sd = safetensors_load_file(lora_path)
        print("LoRA keys:", len(sd))

    return pipe


def get_pipeline() -> Tuple[StableDiffusionImg2ImgPipeline, SDConfig]:
    global _PIPE, _CFG
    if _PIPE is None or _CFG is None:
        _CFG = load_config_from_env()
        _PIPE = _load_pipeline(_CFG)
    return _PIPE, _CFG


# ============================
# Image helpers
# ============================

def _preprocess_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return ImageOps.fit(
        img, (512, 512), method=Image.LANCZOS, centering=(0.5, 0.35)
    )


# ============================
# Main inference
# ============================

def simpsonify_image_bytes(
    image_bytes: bytes,
    prompt: str,
    negative_prompt: Optional[str] = None,
    use_lora: bool = True,
) -> bytes:

    pipe, cfg = get_pipeline()

    if use_lora:
        pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[cfg.lora_scale])
    else:
        pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[0.0])

    img = _preprocess_image(image_bytes)

    gen = None
    if cfg.seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    base_prompt = prompt.strip()

    # -------------------------
    # PASS 1 – FORCE YELLOW
    # -------------------------
    pass1_prompt = (
        base_prompt
        + ", bright yellow skin, flat solid colors, no shading, simple shapes, 2D cel animation"
    )

    pass1_neg = (
        negative_prompt
        or "photo, realistic, natural skin, pores, cinematic lighting, depth of field, bokeh, hdr, 3d render"
    )

    img1 = pipe(
        prompt=pass1_prompt,
        negative_prompt=pass1_neg,
        image=img,
        strength=0.55,
        guidance_scale=4.5,
        num_inference_steps=20,
        generator=gen,
    ).images[0]

    # -------------------------
    # PASS 2 – OUTLINES + STYLE
    # -------------------------
    pass2_prompt = (
        base_prompt
        + ", thick black outline, clean lineart, flat colors, simple shapes, 2D cartoon TV show"
    )

    pass2_neg = (
        negative_prompt
        or "photo, realistic, watercolor, painting, soft shading, blur"
    )

    img2 = pipe(
        prompt=pass2_prompt,
        negative_prompt=pass2_neg,
        image=img1,
        strength=0.50,
        guidance_scale=6.0,
        num_inference_steps=40,
        generator=gen,
    ).images[0]

    buf = BytesIO()
    img2.save(buf, format="PNG")
    return buf.getvalue()
