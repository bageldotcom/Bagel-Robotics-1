FROM python:3.10.21-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DS_BUILD_OPS=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build git ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip setuptools wheel packaging
RUN python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements-train.txt /tmp/requirements-train.txt
RUN python -m pip install -r /tmp/requirements-train.txt && python -m pip check
WORKDIR /opt/vpp2-release
# Source snapshots are bound read-only from the release/workspace at launch.
CMD ["python", "--version"]
