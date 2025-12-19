from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.processor import process_text

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import Response

from backend.services.sd_service import simpsonify_image_bytes

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/simpsonify")
async def simpsonify(
    image: UploadFile = File(...),
    prompt: str = Form("..."),
    negative_prompt: str = Form("..."),
    use_lora: int = Form(1),  # <-- NEU
):
    data = await image.read()
    out_png_bytes = simpsonify_image_bytes(
        image_bytes=data,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_lora=bool(use_lora),  # <-- NEU
    )
    return Response(content=out_png_bytes, media_type="image/png")
