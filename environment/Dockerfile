# hash:sha256:04340a26fb355268cfb57398ad2a8cd1a36b6b9fbb395f6a821e6e47cfca683b
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV CELLPOSE_LOCAL_MODELS_PATH=/root/capsule/code

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-decrosstalk-roi-images-library.git@53ee402c705f10cf637384f59cc2d64e8c7f8a6e#egg=aind-ophys-decrosstalk-roi-images-library
