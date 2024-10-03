FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    trimesh omegaconf einops rembg transformers https://github.com/camenduru/wheels/releases/download/tost/torchmcubes-0.1.0-cp311-cp311-linux_x86_64.whl && \
    git clone -b dev https://github.com/camenduru/TripoSR-hf /content/TripoSR && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/TripoSR/resolve/main/model.ckpt -d /content/model -o model.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/TripoSR/raw/main/config.yaml -d /content/model -o config.yaml

COPY ./worker_runpod.py /content/TripoSR/worker_runpod.py
WORKDIR /content/TripoSR
CMD python worker_runpod.py