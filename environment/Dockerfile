# hash:sha256:f191fbef486c87980e191423c9254c30d9df5a862ba128cf1b26925b0b945077
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV PIPELINE_URL=https://codeocean.allenneuraldynamics.org/capsule/6127754/tree
ENV PIPELINE_VERSION=1.0
ENV VERSION=5.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-opencv=4.2.0+dfsg-5 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-ophys-utils==0.0.7 \
    cellpose==2.2.3 \
    h5py==3.10.0 \
    matplotlib==3.7.3 \
    numpy==1.24.4 \
    opencv-python==4.9.0.80 \
    pandas==2.0.3 \
    pydantic==2.8.2 \
    scikit-image==0.21.0 \
    scipy==1.10.1 \
    suite2p==0.14.3 \
    aind-data-schema==1.1.0 \
    aind-log-utils
