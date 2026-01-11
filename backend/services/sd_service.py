# backend/services/sd_service.py
"""
Stable Diffusion img2img + LoRA service (Simpsonify – final tuned).

- Lazy-load pipeline
- CUDA fp16 / CPU-MPS fp32
- Safety checker disabled (local dev)
- LoRA adapter control
- Deterministic seeding
- Optional two-pass img2img (configured centrally in backend/settings.py)
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file as safetensors_load_file

from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import is_peft_available

from backend.settings import SimpsonifySettings, load_settings


_PIPE: Optional[StableDiffusionImg2ImgPipeline] = None
_SETTINGS: Optional[SimpsonifySettings] = None

_ADAPTER_NAME = "simpsons"


def _load_pipeline(cfg: SimpsonifySettings) -> StableDiffusionImg2ImgPipeline:
    # CPU/MPS -> float32 ist stabil
    # CUDA -> float16
    dtype = torch.float16 if cfg.device == "cuda" else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    )

    pipe = pipe.to(cfg.device)
    # Scheduler (wie in deiner ursprünglichen Version)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # CPU RAM/Speed helper
    pipe.enable_attention_slicing()

    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    lora_path = Path(cfg.lora_path).expanduser().resolve()
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    # diffusers: je nach Version wird entweder file path oder directory akzeptiert
    pipe.load_lora_weights(str(lora_path), adapter_name=_ADAPTER_NAME)

    pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[cfg.lora_scale])

    if cfg.debug:
        print("\n=== PIPE INIT ===")
        print("DEVICE:", cfg.device)
        print("DTYPE:", dtype)
        print("BASE:", cfg.base_model)
        print("LORA:", lora_path)
        print("PEFT available:", is_peft_available())

        try:
            sd = safetensors_load_file(lora_path)
            print("LoRA keys:", len(sd))
        except Exception as e:
            print("Could not read LoRA keys:", e)

    return pipe


def get_pipeline() -> Tuple[StableDiffusionImg2ImgPipeline, SimpsonifySettings]:
    global _PIPE, _SETTINGS
    if _PIPE is None or _SETTINGS is None:
        _SETTINGS = load_settings()
        _PIPE = _load_pipeline(_SETTINGS)
    return _PIPE, _SETTINGS


def _preprocess_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return ImageOps.fit(img, (512, 512), method=Image.LANCZOS, centering=(0.5, 0.35))


def _make_generator(seed: Optional[int], device: str) -> Optional[torch.Generator]:
    if seed is None:
        return None

    # Diffusers akzeptiert Generator; auf CPU ist das am stabilsten.
    # MPS Generator ist nicht überall konsistent implementiert.
    if device == "cuda":
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def simpsonify_image_bytes(
    image_bytes: bytes,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    use_lora: bool = True,
) -> bytes:
    pipe, cfg = get_pipeline()

    # enable/disable adapter
    pipe.set_adapters(
        [_ADAPTER_NAME], adapter_weights=[cfg.lora_scale if use_lora else 0.0]
    )

    img = _preprocess_image(image_bytes)
    gen = _make_generator(cfg.seed, cfg.device)

    base_prompt = (prompt or cfg.default_prompt).strip()
    base_negative = (negative_prompt or cfg.default_negative_prompt).strip()

    if not cfg.two_pass_enabled:
        out = pipe(
            prompt=base_prompt,
            negative_prompt=base_negative,
            image=img,
            strength=cfg.pass2.strength,
            guidance_scale=cfg.pass2.guidance_scale,
            num_inference_steps=cfg.pass2.steps,
            generator=gen,
        ).images[0]
    else:
        # PASS 1
        pass1_prompt = f"{base_prompt}, {cfg.pass1.prompt_suffix}"
        img1 = pipe(
            prompt=pass1_prompt,
            negative_prompt=cfg.pass1.negative_prompt,
            image=img,
            strength=cfg.pass1.strength,
            guidance_scale=cfg.pass1.guidance_scale,
            num_inference_steps=cfg.pass1.steps,
            generator=gen,
        ).images[0]

        # PASS 2
        pass2_prompt = f"{base_prompt}, {cfg.pass2.prompt_suffix}"
        # If you want user negative_prompt to override pass2 negative:
        # pass2_neg = base_negative if negative_prompt else cfg.pass2.negative_prompt
        pass2_neg = cfg.pass2.negative_prompt or base_negative

        out = pipe(
            prompt=pass2_prompt,
            negative_prompt=pass2_neg,
            image=img1,
            strength=cfg.pass2.strength,
            guidance_scale=cfg.pass2.guidance_scale,
            num_inference_steps=cfg.pass2.steps,
            generator=gen,
        ).images[0]

    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
