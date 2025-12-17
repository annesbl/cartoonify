from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from PIL import Image
import io
import os
from dotenv import load_dotenv


from pipeline import SimpsonifyPipeline

load_dotenv()

app = FastAPI(title="Simpsonify SDXL LoRA API")

# --- CONFIG ---
BASE_MODEL = os.environ["SDXL_BASE_MODEL"]
LORA_PATH = os.environ["LORA_PATH"]

DEFAULT_PROMPT = (
    "<lora:simpsonify_e08:1.0>, portrait photo, simpsons cartoon style, "
    "bold black outlines, flat colors, clean cel shading, yellow skin tone, "
    "high quality, consistent face"
)
DEFAULT_NEG = (
    "low quality, blurry, deformed face, extra limbs, bad anatomy, text, watermark"
)

pipe = None

@app.on_event("startup")
def _load():
    global pipe
    pipe = SimpsonifyPipeline(
        base_model=BASE_MODEL,
        lora_path=LORA_PATH,
    )

@app.post("/simpsonify")
async def simpsonify(
    image: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    negative_prompt: str = Form(DEFAULT_NEG),
    strength: float = Form(0.55),
    steps: int = Form(25),
    guidance: float = Form(6.5),
    lora_scale: float = Form(1.0),
    seed: int | None = Form(None),
):
    img_bytes = await image.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    out = pipe.run(
        image=pil,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=float(strength),
        steps=int(steps),
        guidance=float(guidance),
        lora_scale=float(lora_scale),
        seed=seed,
    )

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
