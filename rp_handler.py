import runpod
import os
import sys
import logging
import torch
import tempfile
import requests
import cloudinary
import cloudinary.uploader
from PIL import Image
from io import BytesIO
from omegaconf import OmegaConf

# Cloudinary config
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# ============================================
# LOAD MODEL AT STARTUP
# ============================================
logging.info("Loading OVI 1.1...")

from ovi.ovi_fusion_engine import OviFusionEngine

config = OmegaConf.load("ovi/configs/inference/inference_fusion.yaml")

# 24GB VRAM settings
config.fp8 = True
config.cpu_offload = True
config.ckpt_dir = "./ckpts"
config.output_dir = "/tmp/outputs"
config.video_frame_height_width = [960, 960]

torch.cuda.set_device(0)
ovi_engine = OviFusionEngine(config=config, device=0, target_dtype=torch.bfloat16)
logging.info("OVI engine loaded!")

# ============================================
# HELPERS
# ============================================
def download_image(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    temp_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    img.save(temp_path)
    return temp_path

def upload_to_cloudinary(path):
    result = cloudinary.uploader.upload(path, resource_type="video", folder="ovi_outputs")
    return result["secure_url"]

# ============================================
# HANDLER
# ============================================
def handler(job):
    try:
        inp = job["input"]
        
        prompt = inp.get("prompt", "")
        image_url = inp.get("image_url")
        mode = inp.get("mode", "t2v")
        height = inp.get("height", 960)
        width = inp.get("width", 960)
        seed = inp.get("seed", 100)
        sample_steps = inp.get("sample_steps", 50)
        video_guidance = inp.get("video_guidance_scale", 4.0)
        audio_guidance = inp.get("audio_guidance_scale", 3.0)
        
        image_path = None
        if mode == "i2v":
            if not image_url:
                return {"error": "image_url required for i2v", "status": "failed"}
            image_path = download_image(image_url)
            logging.info(f"Downloaded image: {image_path}")
        
        logging.info(f"Generating video: mode={mode}, {height}x{width}")
        
        video, audio, _ = ovi_engine.generate(
            text_prompt=prompt,
            image_path=image_path,
            video_frame_height_width=[height, width],
            seed=seed,
            solver_name="unipc",
            sample_steps=sample_steps,
            shift=5.0,
            video_guidance_scale=video_guidance,
            audio_guidance_scale=audio_guidance,
            slg_layer=11,
            video_negative_prompt="jitter, bad hands, blur, distortion",
            audio_negative_prompt="robotic, muffled, echo, distorted"
        )
        
        from ovi.utils.io_utils import save_video
        os.makedirs("/tmp/outputs", exist_ok=True)
        out_path = f"/tmp/outputs/ovi_{seed}.mp4"
        save_video(out_path, video, audio, fps=24, sample_rate=16000)
        
        video_url = upload_to_cloudinary(out_path)
        logging.info(f"Uploaded: {video_url}")
        
        os.remove(out_path)
        if image_path:
            os.remove(image_path)
        
        return {"video_url": video_url, "status": "success"}
    
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"error": str(e), "status": "failed"}

runpod.serverless.start({"handler": handler})
