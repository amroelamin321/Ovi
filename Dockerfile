FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 wget

# Copy repo
COPY . /workspace/Ovi/
WORKDIR /workspace/Ovi

# Install correct PyTorch build (CUDA 12.4)
RUN pip install --no-cache-dir torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core OVI deps
RUN pip install --no-cache-dir \
  transformers diffusers accelerate huggingface-hub tokenizers sentencepiece timm \
  optimum optimum-quanto omegaconf einops safetensors

# Audio/video deps
RUN pip install --no-cache-dir \
  librosa soundfile scipy pydub opencv-python av moviepy==1.0.3 open-clip-torch ftfy pandas

# API deps
RUN pip install --no-cache-dir cloudinary requests Pillow numpy fastapi tqdm

# Pre-download 10s 960x960 model to /workspace/ckpts (faster cold start)
RUN python download_weights.py --output-dir /workspace/ckpts --models 960x960_10s || echo "Model will download on first request"

ENV PYTHONUNBUFFERED=1
