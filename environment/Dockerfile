# hash:sha256:a34e962100b76aca06821060645db5fbab2d2f0b7b213ebcb4b8b049953d8551
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV PIPELINE_URL=https://codeocean.allenneuraldynamics.org/capsule/6127754/tree
ENV PIPELINE_VERSION=1.0
ENV VERSION=5.0
ENV CELLPOSE_LOCAL_MODELS_PATH=/root/capsule

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.1.0 \
    aind-log-utils==0.2.1 \
    aind-ophys-utils==0.0.7 \
    aind-qcportal-schema==0.3.2 \
    cellpose==2.2.3 \
    h5py==3.10.0 \
    matplotlib==3.7.3 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scikit-image==0.21.0 \
    suite2p==0.14.3
