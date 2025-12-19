from pathlib import Path

# WICHTIG: Stelle sicher, dass dieses Base-Model zu deinem LoRA passt!
# Beispiele:
# SDXL: "stabilityai/stable-diffusion-xl-base-1.0"
# SD 1.5: "runwayml/stable-diffusion-v1-5"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Dein LoRA hier ablegen:
APP_DIR = Path(__file__).parent
LORA_PATH = APP_DIR /"backend" / "models" / "simpsons_style_lora-000008.safetensors"

# Output
OUT_DIR = APP_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Default Prompt (kannst du später in GUI anpassbar machen)
DEFAULT_PROMPT = "simpsons style, clean lineart, flat colors, cartoon, high quality"
DEFAULT_NEGATIVE = "blurry, low quality, distorted, extra fingers, deformed"
