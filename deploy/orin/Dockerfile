
# 受电弓网故障监测系统 - Jetson AGX Orin 边缘部署 Dockerfile (v2.0)
# 基于 dustynv/transformers:nvgpt-r35.3.1 官方 Jetson 大模型生态镜像构建
# ==============================================================================

# ==============================================================================
FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin

# ==============================================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# 1. 安装 FFmpeg、中文字体及多媒体系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-wqy-zenhei \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装业务系统 Python 依赖（采用 --no-deps 确保绝不触碰底层的 GPU PyTorch 与 OpenCV）
RUN pip3 install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    openpyxl \
    Pillow \
    pyyaml \
    tqdm \
    pandas \
    matplotlib \
    seaborn && \
    pip3 install --no-cache-dir --no-deps \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    ultralytics

WORKDIR /app
COPY . /app




