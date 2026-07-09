FROM python:3.12-slim

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git git-lfs \
    && rm -rf /var/lib/apt/lists/*

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

## SleepFM weights are copied from repo via COPY ./ /challenge above
## Make sure sleepfm/checkpoints/model_base/best.pt exists in your repo
## Use: git lfs track "sleepfm/checkpoints/model_base/best.pt"

ENV PYTHONPATH=/challenge:$PYTHONPATH
