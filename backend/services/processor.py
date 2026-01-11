from io import BytesIO
from PIL import Image


def process_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Bitte Text eingeben."
    return f"Backend hat empfangen: {text}"


def passthrough_image_png(image_bytes: bytes) -> bytes:
    """
    Dummy: liest das Bild, konvertiert zu PNG und gibt es zurück.
    (Später ersetzt du das durch deine echte Simpsonify-Pipeline.)
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
