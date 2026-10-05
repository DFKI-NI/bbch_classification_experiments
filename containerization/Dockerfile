#FROM nvcr.io/nvidia/pytorch:25.05-py3
FROM huggingface/transformers-pytorch-gpu:latest

RUN useradd -u 1000 --create-home --shell /bin/bash docker && echo "docker:docker" | chpasswd && adduser docker sudo

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch downgrade
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# LibMTL version: 1.1.5
RUN pip3 install git+https://github.com/median-research-group/LibMTL.git

COPY python_requirements.txt /tmp/python_requirements.txt
RUN pip3 install -r /tmp/python_requirements.txt

CMD ["/bin/bash"]
