import os
import io
import json
import tempfile
from pathlib import Path

import requests
from PIL import Image

import torch
from omegaconf import OmegaConf

import cloudinary
import cloudinary.uploader

from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.io_utils import save_video

# --- Cloudinary config ---
cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"],
)

ENGINE = None


def load_engine():
    global ENGINE
    if ENGINE is not None:
        return ENGINE

    ckpt_dir = Path("/workspace/ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Ensure 960x960_10s weights exist
    model_file = ckpt_dir / "Ovi" / "model_960x960_10s.safetensors"
    if not model_file.exists():
        from ovi.scripts.download_weights import download_model_weights
        download_model_weights("960x960_10s", str(ckpt_dir))

    config = OmegaConf.load("ovi/configs/inference/inference_fusion.yaml")
    config.ckpt_dir = str(ckpt_dir)
    config.model_name = "960x960_10s"
    config.sp_size = 1
    config.cpu_offload = False
    config.fp8 = False

    ENGINE = OviFusionEngine(
        config=config,
        device=0,
        target_dtype=torch.bfloat16,
    )
    return ENGINE


def generate_video(prompt: str, image_url: str | None, seed: int = 42) -> dict:
    engine = load_engine()

    kwargs = dict(
        text_prompt=prompt,
        video_frame_height_width=[960, 960],
        seed=seed,
        solver_name="unipc",
        sample_steps=50,
        shift=5.0,
        video_guidance_scale=4.0,
        audio_guidance_scale=3.0,
        slg_layer=11,
        video_negative_prompt="jitter, bad hands, blur",
        audio_negative_prompt="robotic, muffled",
    )

    temp_image_path = None
    if image_url:
        resp = requests.get(image_url)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            temp_image_path = f.name
        kwargs["image_path"] = temp_image_path
    else:
        kwargs["image_path"] = None

    video, audio, _ = engine.generate(**kwargs)

    out_path = f"/tmp/ovi_{seed}.mp4"
    save_video(out_path, video, audio, fps=24, sample_rate=16000)

    result = cloudinary.uploader.upload_large(
        out_path,
        resource_type="video",
        folder="ovi_outputs",
        public_id=f"ovi_{seed}",
    )

    if temp_image_path and os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    if os.path.exists(out_path):
        os.remove(out_path)

    return {
        "video_url": result["secure_url"],
        "duration": 10,
        "resolution": "960x960",
        "seed": seed,
    }


def handler(event):
    """
    RunPod serverless entrypoint.
    Expects: {"input": {"prompt": "...", "image_url": "...", "seed": 42}}
    """
    try:
        payload = event.get("input") or {}
        prompt = payload.get("prompt")
        image_url = payload.get("image_url")
        seed = int(payload.get("seed", 42))

        if not prompt:
            return {"error": "prompt is required"}

        return generate_video(prompt, image_url, seed)

    except Exception as e:
        return {"error": str(e)}
