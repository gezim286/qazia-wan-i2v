FROM runpod/base:0.6.2-cuda12.1.0

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod==1.7.9 \
    torch==2.2.2+cu121 \
    torchvision==0.17.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    diffusers==0.31.0 \
    transformers==4.46.3 \
    accelerate==1.0.1 \
    sentencepiece==0.2.0 \
    Pillow==10.4.0 \
    imageio==2.35.1 \
    imageio-ffmpeg==0.5.1 \
    requests==2.32.3

RUN pip install --no-cache-dir \
    git+https://github.com/Wan-Video/Wan2.1.git

COPY handler.py .

CMD ["python", "-u", "handler.py"]
