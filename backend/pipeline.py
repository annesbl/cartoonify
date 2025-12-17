import os
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

class SimpsonifyPipeline:
    def __init__(
        self,
        base_model: str,
        lora_path: str,
        device: str = "cuda",
        dtype=torch.float16,
    ):
        self.device = device
        self.dtype = dtype

        # SDXL img2img pipeline
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            base_model,
            torch_dtype=dtype,
            variant="fp16",
        ).to(device)

        # Optional: speed/memory optimizations
        self.pipe.enable_vae_slicing()
        self.pipe.enable_xformers_memory_efficient_attention() if hasattr(self.pipe, "enable_xformers_memory_efficient_attention") else None

        # Load LoRA
        if not os.path.isfile(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")

        self.pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path),
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
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Set LoRA scale (diffusers supports this pattern)
        if hasattr(self.pipe, "set_adapters"):
            # For newer diffusers multi-adapter setups (optional)
            pass
        # Most common:
        self.pipe.fuse_lora(lora_scale=lora_scale)

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]

        # Unfuse so scale changes work per request
        self.pipe.unfuse_lora()

        return out
