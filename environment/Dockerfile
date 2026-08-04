# hash:sha256:146afeb00af7c237b12fefade0605e9954b8907d3122720a9bc21f73ddbe1441
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV CELLPOSE_LOCAL_MODELS_PATH=/root/capsule/code

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-decrosstalk-roi-images-library.git@7901c88#egg=aind-ophys-decrosstalk-roi-images-library
