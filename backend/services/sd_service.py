# backend/services/sd_service.py
"""
Stable Diffusion img2img + LoRA service (macOS-friendly).

Key properties:
- Lazy-loads the pipeline once (first request).
- macOS/MPS stability: uses float32 (avoids blocky/NaN artifacts common with fp16 on MPS).
- Disables Safety Checker (prevents black images due to false positives) for local dev.
- Robust LoRA activation via adapter_name + set_adapters (no guessing).
- Deterministic seeding without using torch.Generator(device="mps") (often unsupported).
- Simple, reliable preprocessing: center-crop to square and resize to 512x512.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple

import torch
from PIL import Image, ImageOps

from diffusers import StableDiffusionImg2ImgPipeline


# ----------------------------
# Config
# ----------------------------


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

_ADAPTER_NAME = "simpsons"


def _pick_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    # Prefer MPS on Apple Silicon if available
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config_from_env() -> SDConfig:
    base_model = os.getenv("SD_BASE_MODEL", "runwayml/stable-diffusion-v1-5")
    lora_path = os.getenv(
        "SD_LORA_PATH", "./backend/models/simpsons_style_lora-000008.safetensors"
    )

    device = _pick_device(os.getenv("SD_DEVICE"))

    guidance = float(os.getenv("SD_GUIDANCE", "7.0"))
    strength = float(os.getenv("SD_STRENGTH", "0.28"))
    steps = int(os.getenv("SD_STEPS", "20"))

    seed_raw = os.getenv("SD_SEED", "").strip()
    seed = int(seed_raw) if seed_raw else None

    lora_scale = float(os.getenv("SD_LORA_SCALE", "1.6"))
    debug = os.getenv("SD_DEBUG", "0").strip() == "1"

    return SDConfig(
        base_model=base_model,
        lora_path=lora_path,
        device=device,
        guidance_scale=guidance,
        strength=strength,
        num_inference_steps=steps,
        seed=seed,
        lora_scale=lora_scale,
        debug=debug,
    )


# ----------------------------
# Pipeline load / cache
# ----------------------------


def _load_pipeline(cfg: SDConfig) -> StableDiffusionImg2ImgPipeline:
    # macOS/MPS stability: always float32
    dtype = torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        safety_checker=None,  # prevents black images due to false positives
        feature_extractor=None,
    )

    pipe = pipe.to(cfg.device)

    # Prefer stable VAE slicing
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    # LoRA
    # Explicit adapter name + activation (robust)
    # --- LoRA / attn_procs loading (robust fallback)

    lora_scale = float(os.getenv("SD_LORA_SCALE", "1.0"))

    loaded = False

    # Try modern PEFT-style first
    try:
        pipe.load_lora_weights(cfg.lora_path, adapter_name=_ADAPTER_NAME)
        try:
            pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[lora_scale])
        except Exception:
            pass
        loaded = True
    except Exception as e:
        if cfg.debug:
            print("WARN load_lora_weights failed:", e)

    # Fallback: classic attn procs
    if not loaded:
        pipe.unet.load_attn_procs(cfg.lora_path)
        loaded = True

    # Verify injection
    if cfg.debug:
        ap = next(iter(pipe.unet.attn_processors.values()))
        print("ATTN_PROCESSOR_TYPE:", type(ap))

    return pipe


def get_pipeline() -> Tuple[StableDiffusionImg2ImgPipeline, SDConfig]:
    global _PIPE, _CFG
    if _PIPE is None or _CFG is None:
        cfg = load_config_from_env()
        _PIPE = _load_pipeline(cfg)
        _CFG = cfg
    return _PIPE, _CFG


# ----------------------------
# Image preprocessing
# ----------------------------


def _preprocess_image(image_bytes: bytes) -> Image.Image:
    """
    Center-crop to square and resize to 512x512 (SD1.5 sweet spot).
    centering y=0.35 biases crop upward (better for faces in webcam frames).
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.fit(img, (512, 512), method=Image.LANCZOS, centering=(0.5, 0.35))
    return img


# ----------------------------
# Main inference
# ----------------------------


def simpsonify_image_bytes(
    image_bytes: bytes,
    prompt: str,
    negative_prompt: Optional[str] = None,
    use_lora: bool = True,
) -> bytes:
    pipe, cfg = get_pipeline()

    # (Optional) switch LoRA on/off per call (for A/B testing)
    if use_lora:
        try:
            pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[cfg.lora_scale])
        except Exception:
            pass
    else:
        try:
            pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[0.0])
        except Exception:
            pass

    img = _preprocess_image(image_bytes)

    # Deterministic seeding: avoid torch.Generator(device="mps") (often unsupported)
    gen = None
    if cfg.seed is not None:
        if cfg.device == "mps":
            gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        else:
            gen = torch.Generator(device=cfg.device).manual_seed(int(cfg.seed))

    # Run img2img
    result = pipe(
        prompt=prompt,
        image=img,
        strength=float(cfg.strength),
        guidance_scale=float(cfg.guidance_scale),
        num_inference_steps=int(cfg.num_inference_steps),
        negative_prompt=negative_prompt,
        generator=gen,
    )

    out = result.images[0]
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
