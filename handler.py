import os
import io
import sys
import json
import torch
import requests
import tempfile
import traceback
from pathlib import Path

# Add OVI to path
sys.path.insert(0, '/workspace/Ovi')

import cloudinary
import cloudinary.uploader
from omegaconf import OmegaConf
from PIL import Image
from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.io_utils import save_video

# Cloudinary setup
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
)

# Global engine (loaded once)
ENGINE = None

def download_and_setup_model():
    """Download model if not present"""
    ckpt_path = Path("/workspace/ckpts/Ovi/model_960x960_10s.safetensors")
    
    if not ckpt_path.exists():
        print("📥 Downloading OVI 1.1 10-second model (11GB)...")
        os.chdir("/workspace/Ovi")
        import subprocess
        result = subprocess.run([
            "python3", "download_weights.py",
            "--output-dir", "/workspace/ckpts",
            "--models", "960x960_10s"
        ], capture_output=True, text=True)
        
        print("STDOUT:", result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            raise RuntimeError("Model download failed")
    
    print("✅ Model ready at:", ckpt_path)
    return "/workspace/ckpts"

def load_engine():
    """Initialize OVI Fusion Engine (loaded once globally)"""
    global ENGINE
    
    if ENGINE is not None:
        return ENGINE
    
    print("🚀 Loading OVI Fusion Engine...")
    
    ckpt_dir = download_and_setup_model()
    
    # Load config
    config = OmegaConf.load("/workspace/Ovi/ovi/configs/inference/inference_fusion.yaml")
    
    # Critical settings for 10-second generation
    config.ckpt_dir = ckpt_dir
    config.model_name = "960x960_10s"  # This ensures 10s model (241 frames, 314 audio)
    config.sp_size = 1
    config.cpu_offload = False  # A100 has enough VRAM
    config.fp8 = False  # Full precision on A100

    ENGINE = OviFusionEngine(
        config=config,
        device=0,
        target_dtype=torch.bfloat16
    )
    
    print("✅ Engine loaded successfully!")
    return ENGINE

def handler(job):
    """RunPod serverless handler"""
    try:
        job_input = job.get("input", {})
        
        prompt = job_input.get("prompt")
        image_url = job_input.get("image_url")
        seed = job_input.get("seed", 42)
        
        if not prompt:
            return {"error": "Missing 'prompt' in input"}
        
        mode = "i2v" if image_url else "t2v"
        print(f"\n🎬 Generating {mode} (10s, 960×960)")
        print(f"📝 Prompt: {prompt[:100]}...")
        
        # Load engine
        engine = load_engine()
        
        # Prepare generation arguments
        gen_args = {
            "text_prompt": prompt,
            "image_path": None,
            "video_frame_height_width": [960, 960],
            "seed": seed,
            "solver_name": "unipc",
            "sample_steps": 50,
            "shift": 5.0,
            "video_guidance_scale": 4.0,
            "audio_guidance_scale": 3.0,
            "slg_layer": 11,
            "video_negative_prompt": "jitter, bad hands, blur",
            "audio_negative_prompt": "robotic, muffled",
        }
        
        # Handle i2v
        if image_url:
            print(f"📥 Downloading image from: {image_url}")
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img.save(f.name)
                gen_args["image_path"] = f.name
                print(f"✅ Image saved to: {f.name}")
        
        # Generate video + audio
        print("⏳ Generating... (2-3 minutes on A100)")
        video, audio, _ = engine.generate(**gen_args)
        
        # Save video
        out_path = f"/tmp/video_{seed}.mp4"
        save_video(out_path, video, audio, fps=24, sample_rate=16000)
        print(f"✅ Video saved to: {out_path}")
        
        # Upload to Cloudinary
        print("☁️ Uploading to Cloudinary...")
        result = cloudinary.uploader.upload_large(
            out_path,
            resource_type="video",
            folder="ovi_outputs",
            public_id=f"ovi_{seed}",
            timeout=300
        )
        
        video_url = result["secure_url"]
        print(f"✅ Video URL: {video_url}")
        
        # Cleanup
        if gen_args["image_path"] and os.path.exists(gen_args["image_path"]):
            os.remove(gen_args["image_path"])
        if os.path.exists(out_path):
            os.remove(out_path)
        
        return {
            "video_url": video_url,
            "duration": 10,
            "resolution": "960x960",
            "seed": seed,
            "cloudinary_id": result["public_id"]
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

# RunPod handler entry
if __name__ == "__main__":
    import runpod
    
    print("🚀 Starting OVI 1.1 RunPod Serverless Handler")
    runpod.serverless.start({"handler": handler})
