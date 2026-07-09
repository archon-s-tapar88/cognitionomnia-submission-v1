FROM python:3.12-slim

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git git-lfs \
    && rm -rf /var/lib/apt/lists/*

## Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

## SleepFM weights are included in the repository using Git LFS:
## sleepfm/checkpoints/model_base/best.pt

ENV PYTHONPATH=/challenge:$PYTHONPATH
