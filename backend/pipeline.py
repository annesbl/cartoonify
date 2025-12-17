import os
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SimpsonifyPipeline:
    def __init__(self, base_model: str, lora_path: str, device: str | None = None):
        self.device = device or pick_device()
        self.dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        if not os.path.isfile(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")

        # For SD1.5, fp16 variant is not a thing we need to force; keep it simple/robust.
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            base_model,
            torch_dtype=self.dtype,
        ).to(self.device)

        # Safer slicing call across versions:
        try:
            self.pipe.vae.enable_slicing()
        except Exception:
            pass

        # xformers typically not available on macOS
        if self.device == "cuda" and hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            self.pipe.enable_xformers_memory_efficient_attention()

        # Load LoRA
        self.pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path),
            adapter_name="default_0",
        )

    @torch.inference_mode()
    def run(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        guidance: float,
        lora_scale: float,
        seed: int | None,
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # Apply LoRA strength per request (if supported)
        if hasattr(self.pipe, "fuse_lora"):
            self.pipe.fuse_lora(lora_scale=float(lora_scale))

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
        ).images[0]

        if hasattr(self.pipe, "unfuse_lora"):
            self.pipe.unfuse_lora()

        return out
