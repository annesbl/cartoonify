from __future__ import annotations

import os
from pathlib import Path

# We call the official diffusers training entry via accelerate.
# This script only prepares paths and a robust VRAM-safe config.

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # --- paths ---
    images_dir = repo_root / "data" / "train_images"
    captions_dir = repo_root / "data" / "train_captions"
    output_dir = repo_root / "output" / "lora" / "dsai_simpson_style_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise SystemExit(f"Missing images: {images_dir}")
    if not captions_dir.exists():
        raise SystemExit(f"Missing captions: {captions_dir} (run scripts/01_make_captions.py first)")

    # --- config (8GB VRAM safe) ---
    pretrained_model = "runwayml/stable-diffusion-v1-5"
    resolution = 512
    train_batch_size = 1
    grad_accum = 8
    max_train_steps = 3000
    lr = 1e-4
    rank = 16

    # IMPORTANT: we store caption files separately; diffusers expects image+caption pairing.
    # We'll create a temporary "combined" view by symlinking captions next to images.
    # To avoid touching the dataset, we create a working directory.
    work_dir = repo_root / "data" / "_train_workdir"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks: image + matching caption side-by-side in work_dir
    # (Diffusers training scripts commonly expect this pattern)
    import shutil

    # clear work_dir
    for p in work_dir.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink()

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for img in images_dir.iterdir():
        if img.suffix.lower() not in exts:
            continue
        cap = captions_dir / f"{img.stem}.txt"
        if not cap.exists():
            raise SystemExit(f"Missing caption for {img.name}: expected {cap.name}")

        # symlink/copy fallback
        img_link = work_dir / img.name
        cap_link = work_dir / cap.name
        try:
            img_link.symlink_to(img)
            cap_link.symlink_to(cap)
        except Exception:
            # if symlink not permitted in this container, copy instead
            shutil.copy2(img, img_link)
            shutil.copy2(cap, cap_link)

    print(f"Prepared workdir: {work_dir}")

    # Build the accelerate command
    cmd = (
        "accelerate launch -m diffusers.examples.text_to_image.train_text_to_image_lora "
        f'--pretrained_model_name_or_path="{pretrained_model}" '
        f'--train_data_dir="{work_dir}" '
        f"--resolution={resolution} "
        f"--train_batch_size={train_batch_size} "
        f"--gradient_accumulation_steps={grad_accum} "
        f"--checkpointing_steps=500 "
        f'--learning_rate="{lr}" '
        f'--lr_scheduler="constant" '
        f"--max_train_steps={max_train_steps} "
        f"--rank={rank} "
        f'--mixed_precision="fp16" '
        f"--gradient_checkpointing "
        f"--enable_xformers_memory_efficient_attention "
        f'--validation_prompt="dsai_simpson_style, portrait, yellow skin, flat colors, cartoon tv animation" '
        f"--validation_steps=250 "
        f'--output_dir="{output_dir}"'
    )

    print("\nRunning:\n", cmd, "\n")
    os.system(cmd)


if __name__ == "__main__":
    main()
