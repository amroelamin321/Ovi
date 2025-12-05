import os
import json
import subprocess
import tempfile
import requests
from pathlib import Path
import cloudinary
import cloudinary.uploader
from PIL import Image
import io

# Cloudinary config
cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"]
)

def handler(event):
    try:
        input_data = event["input"]
        prompt = input_data.get("prompt")
        image_url = input_data.get("image_url")
        seed = input_data.get("seed", 42)
        
        if not prompt:
            return {"error": "prompt required"}
        
        print(f"🚀 OVI 1.1 ({'i2v' if image_url else 't2v'}) - {prompt[:50]}")
        
        # Create temp config file (OVI inference.py reads YAML)
        config_path = "/tmp/inference_config.yaml"
        mode = "i2v" if image_url else "t2v"
        
        config_content = f"""
output_dir: /tmp
ckpt_dir: /workspace/ckpts
model_name: 960x960_10s
sp_size: 1
cpu_offload: false
fp8: false
text_prompt: "{prompt}"
mode: {mode}
video_frame_height_width: [960, 960]
seed: {seed}
solver_name: unipc
sample_steps: 50
shift: 5.0
video_guidance_scale: 4.0
audio_guidance_scale: 3.0
slg_layer: 11
video_negative_prompt: "jitter, bad hands, blur"
audio_negative_prompt: "robotic, muffled"
"""
        
        with open(config_path, "w") as f:
            f.write(config_content)
        
        # i2v: download image to /tmp/input.jpg
        image_path = None
        if image_url:
            resp = requests.get(image_url)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            image_path = "/tmp/input.jpg"
            img.save(image_path)
        
        # Run OFFICIAL inference.py (exact repo command)
        cmd = [
            "python", "inference.py",
            "--config-file", config_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspace/Ovi")
        
        if result.returncode != 0:
            print("inference.py failed:", result.stderr)
            return {"error": result.stderr}
        
        # Find generated video (inference.py saves to output_dir)
        video_files = list(Path("/tmp").glob("*.mp4"))
        if not video_files:
            return {"error": "No video generated"}
        
        video_path = video_files[0]
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload_large(
            str(video_path),
            resource_type="video",
            folder="ovi_1.1",
            public_id=f"ovi_{seed}"
        )
        
        # Cleanup
        Path(config_path).unlink(missing_ok=True)
        if image_path:
            Path(image_path).unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)
        
        return {
            "status": "success",
            "video_url": upload_result["secure_url"],
            "duration": 10,
            "resolution": "960x960",
            "seed": seed
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}
