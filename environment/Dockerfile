# hash:sha256:56824b1aa230019027f229e03736d91dfa2ee0c5e2eba092401311062e352cba
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV CELLPOSE_LOCAL_MODELS_PATH=/root/capsule/code

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-decrosstalk-roi-images-library.git@b13e3f2#egg=aind-ophys-decrosstalk-roi-images-library
