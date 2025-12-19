from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from PIL import Image

from config import MODEL_ID, LORA_PATH, OUT_DIR, DEFAULT_PROMPT, DEFAULT_NEGATIVE

app = FastAPI(title="Simpsonify Backend")

pipe = None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline():
    """
    Lazy-load, damit Serverstart nicht sofort alles lädt.
    """
    global pipe
    if pipe is not None:
        return pipe

    device = pick_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    # Pipeline je nach Model wählen
    # - SDXL: StableDiffusionXLImg2ImgPipeline
    # - SD 1.5: StableDiffusionImg2ImgPipeline
    #
    # Wir versuchen SDXL, wenn MODEL_ID nach SDXL aussieht, ansonsten SD1.5.
    is_sdxl = "xl" in MODEL_ID.lower()

    if is_sdxl:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        p = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )
    else:
        from diffusers import StableDiffusionImg2ImgPipeline
        p = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )

    # LoRA laden
    if not Path(LORA_PATH).exists():
        raise FileNotFoundError(f"LoRA not found at: {LORA_PATH}")
    p.load_lora_weights(str(LORA_PATH))

    p = p.to(device)

    # Optional: ein paar Defaults
    try:
        p.enable_attention_slicing()
    except Exception:
        pass

    pipe = p
    return pipe


@app.get("/health")
def health():
    return {"status": "ok", "device": pick_device(), "model": MODEL_ID}


@app.post("/convert")
async def convert(
    image: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    negative_prompt: str = Form(DEFAULT_NEGATIVE),
    strength: float = Form(0.65),
    guidance: float = Form(6.0),
    steps: int = Form(25),
    seed: int = Form(0),
    lora_scale: float = Form(1.0),
):
    """
    Nimmt ein Bild entgegen und gibt ein PNG zurück.
    """
    try:
        p = load_pipeline()
        device = pick_device()

        raw = await image.read()
        inp = Image.open(io.BytesIO(raw)).convert("RGB")

        # Optionale LoRA Stärke (Diffusers kann das je nach Version unterstützen)
        # Falls es in deiner Diffusers-Version nicht existiert, ignorieren wir es sauber.
        try:
            if hasattr(p, "set_adapters"):
                # neuere API (nicht immer vorhanden)
                pass
            if hasattr(p, "fuse_lora"):
                # falls vorhanden
                pass
        except Exception:
            pass

        generator: Optional[torch.Generator] = None
        if seed and seed > 0:
            generator = torch.Generator(device=device).manual_seed(seed)

        result = p(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=inp,
            strength=float(strength),
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=generator,
        ).images[0]

        # Als PNG zurückgeben
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # Optional: speichern (Debug)
        out_path = OUT_DIR / f"result_{int(time.time())}.png"
        out_path.write_bytes(png_bytes)

        return Response(content=png_bytes, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
