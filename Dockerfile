FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 wget \
    && rm -rf /var/lib/apt/lists/*

# Copy repo files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir runpod cloudinary requests Pillow
RUN pip install --no-cache-dir -r requirements.txt

# Flash attention (optional)
RUN pip install flash_attn --no-build-isolation || true

# Download weights (includes 10s model from latest repo)
RUN python3 download_weights.py

# Download FP8 weights for 24GB VRAM
RUN wget -O "./ckpts/Ovi/model_fp8_e4m3fn.safetensors" \
    "https://huggingface.co/rkfg/Ovi-fp8_quantized/resolve/main/model_fp8_e4m3fn.safetensors"

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "rp_handler.py"]
