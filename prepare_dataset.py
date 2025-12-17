from PIL import Image
import os

SRC_DIR = "raw_images"
DST_DIR = "prepared_images"
TARGET_SIZE = 1024

os.makedirs(DST_DIR, exist_ok=True)

count = 0

for fn in os.listdir(SRC_DIR):
    if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    try:
        img = Image.open(os.path.join(SRC_DIR, fn)).convert("RGB")

        # quadratisch machen (Center Crop)
        w, h = img.size
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        img = img.crop((left, top, left + min_side, top + min_side))

        # auf SDXL-Format skalieren
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

        img.save(os.path.join(DST_DIR, fn))
        count += 1

    except Exception as e:
        print(f"Skipping {fn}: {e}")

print(f"Prepared {count} images.")
