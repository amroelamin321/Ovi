import os
import json
import torch
from pathlib import Path
import cloudinary
import cloudinary.uploader
import tempfile
import requests
from PIL import Image
import io
from omegaconf import OmegaConf
import sys
sys.path.insert(0, '/workspace/Ovi')

from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.io_utils import save_video

# Cloudinary
cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"], 
    api_secret=os.environ["CLOUDINARY_API_SECRET"]
)

def handler(event):
    try:
        input_data = event["input"]
        prompt = input_data.get("prompt")
        image_url = input_data.get("image_url", None)
        seed = input_data.get("seed", 42)
        
        if not prompt:
            return {"error": "Missing prompt"}
        
        print(f"🚀 OVI 1.1 {'i2v' if image_url else 't2v'} - {prompt[:50]}...")
        
        # Load 10s config + model
        config_path = "/workspace/Ovi/ovi/configs/inference/inference_fusion.yaml"
        config = OmegaConf.load(config_path)
        config.ckpt_dir = "/workspace/ckpts"
        config.model_name = "960x960_10s"
        config.sp_size = 1
        config.cpu_offload = False
        config.fp8 = False
        
        engine = OviFusionEngine(
            config=config, 
            device=0, 
            target_dtype=torch.bfloat16
        )
        
        gen_kwargs = {
            "text_prompt": prompt,
            "video_frame_height_width": [960, 960],
            "seed": seed,
            "solver_name": "unipc",
            "sample_steps": 50,
            "shift": 5.0, 
            "video_guidance_scale": 4.0,
            "audio_guidance_scale": 3.0,
            "slg_layer": 11,
            "video_negative_prompt": "jitter, bad hands, blur",
            "audio_negative_prompt": "robotic, muffled"
        }
        
        image_path = None
        if image_url:
            resp = requests.get(image_url)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img.save(f.name)
                image_path = f.name
            gen_kwargs["image_path"] = image_path
        
        # Generate 10s video+audio
        video, audio, _ = engine.generate(**gen_kwargs)
        
        # Save MP4
        out_path = f"/tmp/ovi_{seed}.mp4"
        save_video(out_path, video, audio, fps=24, sample_rate=16000)
        
        # Upload to Cloudinary
        result = cloudinary.uploader.upload_large(
            out_path,
            resource_type="video",
            folder="ovi_1.1_outputs",
            public_id=f"ovi_{seed}"
        )
        
        # Cleanup
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
        os.unlink(out_path)
        
        return {
            "status": "success",
            "video_url": result["secure_url"],
            "duration_seconds": 10,
            "resolution": "960x960", 
            "seed": seed,
            "cloudinary_public_id": result["public_id"]
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}
