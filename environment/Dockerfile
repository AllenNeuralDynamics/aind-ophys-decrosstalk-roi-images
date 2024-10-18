# hash:sha256:e6a9380356c40d1c2942977f5abd6394e412cf3ee6c6d3c6e173229ed82dca81
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV VERSION=3.0
ENV PIPELINE_URL=https://codeocean.allenneuraldynamics.org/capsule/6127754/tree
ENV PIPELINE_VERSION=1.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-opencv=4.2.0+dfsg-5 \
        vim=2:8.1.2269-1ubuntu5.22 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.22.0 \
    aind-ophys-utils==0.0.7 \
    cellpose==2.2.3 \
    h5py==3.10.0 \
    matplotlib==3.7.3 \
    numpy==1.24.4 \
    opencv-python==4.9.0.80 \
    pandas==2.0.3 \
    scikit-image==0.21.0 \
    scipy==1.10.1 \
    suite2p==0.14.3
