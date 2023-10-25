# hash:sha256:33ea20a39ad2f9ef5f23a01c4d679cc971ccf060c3350d844b9f0f441c13917d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        vim=2:8.1.2269-1ubuntu5.20 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    cellpose==2.2.3 \
    h5py==3.10.0 \
    matplotlib==3.7.3 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scikit-image==0.21.0 \
    scipy==1.10.1
