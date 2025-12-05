FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 wget

# Clone OVI 1.1
RUN git clone https://github.com/character-ai/Ovi.git /workspace/Ovi
WORKDIR /workspace/Ovi

# PyTorch 2.6 + CUDA 12.4 (matches OVI requirements)
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# OVI exact deps (from requirements.txt + fixes)
RUN pip install --no-cache-dir \
    transformers \
    diffusers \
    accelerate \
    huggingface-hub \
    tokenizers \
    sentencepiece \
    timm \
    optimum \
    optimum-quanto \
    omegaconf \
    einops \
    safetensors \
    librosa \
    soundfile \
    scipy \
    pydub \
    opencv-python \
    av \
    moviepy==1.0.3 \
    open-clip-torch \
    ftfy \
    pandas \
    cloudinary \
    requests \
    Pillow \
    numpy \
    tqdm

# Install FlashAttention-3 (H100/A100 compatible)
RUN pip install flash-attn --no-build-isolation

# Pre-download 960x960_10s model (11GB)
RUN python download_weights.py --output-dir /workspace/ckpts --models 960x960_10s
