FROM python:3.12-slim

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install system dependencies for LightGBM, scientific computing, and SleepFM
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git git-lfs wget \
    && rm -rf /var/lib/apt/lists/*

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

## Download SleepFM pretrained base model weights during build
## (Internet is available during Docker build, NOT during training/inference)
RUN mkdir -p /challenge/sleepfm/checkpoints/model_base && \
    wget -q -O /challenge/sleepfm/checkpoints/model_base/best.pt \
    "https://github.com/zou-group/sleepfm-clinical/releases/download/v1.0/model_base_best.pt" || \
    echo "WARNING: Could not download SleepFM weights. Ensure weights are in repo."

RUN wget -q -O /challenge/sleepfm/checkpoints/model_base/config.json \
    "https://raw.githubusercontent.com/zou-group/sleepfm-clinical/main/sleepfm/checkpoints/model_base/config.json" || \
    echo "WARNING: Could not download SleepFM config. Ensure config is in repo."

ENV PYTHONPATH=/challenge:$PYTHONPATH
