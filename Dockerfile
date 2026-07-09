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

## Set PYTHONPATH so 'sleepfm' package is discoverable
ENV PYTHONPATH=/challenge

CMD ["bash"]
