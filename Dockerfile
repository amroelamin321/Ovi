FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 wget

# Clone OVI 1.1 (gets ALL files including inference.py)
RUN git clone https://github.com/character-ai/Ovi.git /workspace/Ovi
WORKDIR /workspace/Ovi

# EXACT PyTorch from OVI README
RUN pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install requirements.txt FIRST (official deps)
RUN pip install -r requirements.txt

# FlashAttention-3 (REQUIRED per README)
RUN pip install flash-attn --no-build-isolation

# Additional deps for API + Cloudinary
RUN pip install cloudinary requests pillow

# Pre-download 960x960_10s model (~11GB)
RUN python download_weights.py --output-dir /workspace/ckpts --models 960x960_10s || true
