# hash:sha256:ed2358b0f1842a362cfcca4dba7a0435f4c28f08fe4ec753b0b1f892e048e606
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV CELLPOSE_LOCAL_MODELS_PATH=/root/capsule/code

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    -e https://github.com/AllenNeuralDynamics/aind-ophys-decrosstalk-roi-images-library.git@53ee402c705f10cf637384f59cc2d64e8c7f8a6e#egg=aind-ophys-decrosstalk-roi-images-library
