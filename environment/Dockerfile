# hash:sha256:5712f8e51cf42d4c82b40f5768fe361f4a1cb4c34b5e194f5968c9a08536734e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV CELLPOSE_LOCAL_MODELS_PATH=/root/capsule/code

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-decrosstalk-roi-images-library.git@2207581#egg=aind-ophys-decrosstalk-roi-images-library
