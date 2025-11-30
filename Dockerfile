FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy repo code (NO models - they are on Network Volume)
COPY . /app

# Install RunPod and Cloudinary
RUN pip install --no-cache-dir runpod cloudinary requests Pillow

# Install OVI requirements
RUN pip install --no-cache-dir -r requirements.txt

# Try to install Flash Attention (optional, skip if fails)
RUN pip install flash_attn --no-build-isolation || echo "Flash Attention skipped"

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "rp_handler.py"]
