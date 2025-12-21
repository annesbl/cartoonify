# backend/routes.py
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import Response

from backend.services.sd_service import simpsonify_image_bytes
from backend.settings import load_settings, settings_as_dict

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/config")
def config():
    s = load_settings()
    return settings_as_dict(s)


@router.post("/simpsonify")
async def simpsonify(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    use_lora: int = Form(1),
):
    data = await image.read()

    out_png_bytes = simpsonify_image_bytes(
        image_bytes=data,
        prompt=prompt.strip() or None,
        negative_prompt=negative_prompt.strip() or None,
        use_lora=bool(use_lora),
    )
    return Response(content=out_png_bytes, media_type="image/png")
