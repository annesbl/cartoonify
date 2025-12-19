from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple

import torch
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline

try:
    # Available in diffusers for SDXL img2img
    from diffusers import StableDiffusionXLImg2ImgPipeline
except Exception:
    StableDiffusionXLImg2ImgPipeline = None  # type: ignore


@dataclass
class SDConfig:
    base_model: str
    lora_path: str
    device: str
    use_sdxl: bool
    guidance_scale: float
    strength: float
    num_inference_steps: int
    seed: Optional[int]


_PIPE: Optional[object] = None
_CFG: Optional[SDConfig] = None


def _pick_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for(device: str) -> torch.dtype:
    # macOS MPS: float16 ist oft instabil -> schwarze Bilder / NaNs
    if device == "mps":
        return torch.float32
    return torch.float32


def load_config_from_env() -> SDConfig:
    base_model = os.getenv("SD_BASE_MODEL", "runwayml/stable-diffusion-v1-5")
    lora_path = os.getenv("SD_LORA_PATH", "./lora.safetensors")

    device = _pick_device(os.getenv("SD_DEVICE"))

    # Decide SDXL vs SD1.5:
    # - set SD_USE_SDXL=1 explicitly, OR
    # - infer from model name containing "xl"
    use_sdxl = os.getenv("SD_USE_SDXL", "").strip() == "1" or (
        "xl" in base_model.lower()
    )

    guidance_scale = float(os.getenv("SD_GUIDANCE", "7.0"))
    strength = float(os.getenv("SD_STRENGTH", "0.65"))
    steps = int(os.getenv("SD_STEPS", "25"))
    seed_raw = os.getenv("SD_SEED", "").strip()
    seed = int(seed_raw) if seed_raw else None

    return SDConfig(
        base_model=base_model,
        lora_path=lora_path,
        device=device,
        use_sdxl=use_sdxl,
        guidance_scale=guidance_scale,
        strength=strength,
        num_inference_steps=steps,
        seed=seed,
    )


def _load_pipeline(cfg: SDConfig):

    dtype = _dtype_for(cfg.device)

    if cfg.use_sdxl:
        if StableDiffusionXLImg2ImgPipeline is None:
            raise RuntimeError(
                "SDXL pipeline not available in your diffusers version. Please update diffusers."
            )
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            cfg.base_model,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg.base_model,
            torch_dtype=dtype,
        )

    pipe = pipe.to(cfg.device)
    print("==== DIFFUSION DEBUG ====")
    print("PIPE:", pipe.__class__.__name__)
    print("BASE MODEL:", cfg.base_model)
    print("LORA PATH:", cfg.lora_path)
    print("==========================")
    # Disable safety checker for local dev (prevents black images on false positives)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False

    # Mac-friendly memory tweaks
    pipe.enable_attention_slicing()

    import os

    adapter_name = "simpsons"
    lora_scale = float(os.getenv("SD_LORA_SCALE", "1.0"))

    # --- Load LoRA with an explicit adapter name
    pipe.load_lora_weights(cfg.lora_path, adapter_name=adapter_name)
    ap = next(iter(pipe.unet.attn_processors.values()))
    print("ATTN_PROCESSOR_TYPE:", type(ap))

    # --- Activate adapter (this is the part many people miss)
    try:
        pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
    except Exception as e:
        print("WARN set_adapters failed:", e)

    # --- Fuse LoRA into weights (often makes the effect much stronger & consistent)
    try:
        pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=lora_scale)
    except Exception:
        try:
            pipe.fuse_lora(lora_scale=lora_scale)
        except Exception as e:
            print("WARN fuse_lora failed:", e)

    # Optional speed/memory helpers if present:
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    return pipe


def get_pipeline() -> Tuple[object, SDConfig]:
    global _PIPE, _CFG
    if _PIPE is None or _CFG is None:
        cfg = load_config_from_env()
        _PIPE = _load_pipeline(cfg)
        _CFG = cfg
    return _PIPE, _CFG


def simpsonify_image_bytes(
    image_bytes: bytes,
    prompt: str,
    negative_prompt: str | None = None,
    use_lora: bool = True,
) -> bytes:

    pipe, cfg = get_pipeline()

    adapter_name = "simpsons"
    lora_scale = float(os.getenv("SD_LORA_SCALE", "1.0"))

    if use_lora:
        try:
            pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
        except Exception:
            pass
    else:
        # disable adapters
        try:
            pipe.set_adapters([], adapter_weights=[])
        except Exception:
            # fallback: try to disable by setting scale to 0
            pass

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    # Crop center area (portrait)
    crop_w = int(w * 0.65)
    crop_h = int(h * 0.65)
    left = (w - crop_w) // 2
    top = int((h - crop_h) * 0.25)  # etwas nach oben verschoben
    img = img.crop((left, top, left + crop_w, top + crop_h))

    # Reasonable size for macOS; keep aspect ratio. SDXL likes ~1024, SD1.5 likes ~512.
    target = 1024 if cfg.use_sdxl else 512
    img.thumbnail((target, target), Image.LANCZOS)
    img = img.resize((512, 512))

    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    lora_scale = float(os.getenv("SD_LORA_SCALE", "1.4"))
    # Run img2img
    result = pipe(
        prompt=prompt,
        image=img,
        strength=cfg.strength,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        negative_prompt=negative_prompt,
        generator=generator,
        cross_attention_kwargs={"scale": lora_scale},
    )

    out = result.images[0]
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
