FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/Ovi
WORKDIR /workspace/Ovi

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
    transformers diffusers accelerate optimum optimum-quanto \
    omegaconf einops safetensors \
    librosa soundfile scipy pydub opencv-python av \
    moviepy==1.0.3 open-clip-torch ftfy pandas \
    cloudinary requests Pillow numpy tqdm flask

# Pre-download model to avoid cold start delays (optional - speeds up first run)
RUN mkdir -p /workspace/ckpts && \
    python3 download_weights.py --output-dir /workspace/ckpts --models 960x960_10s || true

COPY handler.py /workspace/handler.py
CMD ["python3", "-u", "/workspace/handler.py"]
