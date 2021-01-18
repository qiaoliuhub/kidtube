# This part is copied form the huggingface/transformers-pytorch-gpu
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
LABEL maintainer="Hugging Face"
LABEL repository="transformers"

RUN apt-get update && \
    apt-get install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch \
    pytorch-lightning \
    pandas \
    scikit-learn \
    wandb

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && \
    python3 setup.py install && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /workspace
RUN git clone https://github.com/huggingface/transformers
RUN cd transformers/ && \
    python3 -m pip install --no-cache-dir .

CMD ["/bin/bash"]