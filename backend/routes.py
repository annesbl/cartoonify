from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import Response

from services.sd_service import simpsonify_image_bytes

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/simpsonify")
async def simpsonify(
    image: UploadFile = File(...),
    prompt: str = Form(
    "simpsons character portrait, thick black outline, clean lineart, flat colors, cel shading, yellow skin, 2D cartoon, simple background"
    ),
    negative_prompt: str = Form(
    "photo, realistic, airbrush, smooth skin, beauty, makeup, cinematic lighting, depth of field, bokeh, detailed skin, pores, hdr, 3d render, blurry"
    ),
    use_lora: int = Form(1),
):
    data = await image.read()
    print("PROMPT_IN:", prompt)
    print("NEG_IN:", negative_prompt)
    print("USE_LORA:", use_lora)

    out_png_bytes = simpsonify_image_bytes(
        image_bytes=data,
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        use_lora=bool(use_lora),
    )
    return Response(content=out_png_bytes, media_type="image/png")
