# Kiln on the AWS Neuron vLLM inference image: Neuron SDK 2.32, Python 3.13, Ubuntu 24.04.
#
# The base already holds everything Kiln imports at the versions Kiln is tested with (torch 2.11,
# libtorch-neuronx-lite, nki 0.6.0, neuronx-cc 2.27, transformers 5.15, tokenizers, safetensors,
# huggingface_hub, fastapi, uvicorn, pydantic, xgrammar, psutil, pytest, the aws CLI), so this layer
# installs Kiln alone, without dependencies, and checks every Neuron piece before the build may finish
# (docker/check_image.py).
#
# Build from the root of a public Kiln tree (github.com/foxl-ai/kiln at the commit being released):
#   docker build -f docker/Dockerfile \
#     --build-arg KILN_VERSION=<kiln/__init__.py __version__> --build-arg KILN_REVISION=<that commit> \
#     -t kiln:<version> .
#
# The base is pinned by digest. Its tag was read from the Neuron image gallery
# (gallery.ecr.aws/neuron/pytorch-inference-vllm-neuronx), its digest from
# `docker pull` / the registry's docker-content-digest header for that tag (2026-10-05).
ARG BASE_IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04
ARG BASE_DIGEST=sha256:be11c204f419a63e2487b2124005156dad091fb9edbfcadf42d81b745e284c12
FROM ${BASE_IMAGE}@${BASE_DIGEST}

ARG BASE_IMAGE
ARG BASE_DIGEST
ARG KILN_VERSION
ARG KILN_REVISION

# `maintainer` is set because the base's own ("Amazon AI") would otherwise be inherited.
LABEL maintainer="Foxl (https://github.com/foxl-ai/kiln)" \
      org.opencontainers.image.title="Kiln" \
      org.opencontainers.image.description="An LLM inference engine for AWS Trainium" \
      org.opencontainers.image.source="https://github.com/foxl-ai/kiln" \
      org.opencontainers.image.url="https://github.com/foxl-ai/kiln" \
      org.opencontainers.image.documentation="https://github.com/foxl-ai/kiln#container" \
      org.opencontainers.image.vendor="Foxl" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${KILN_VERSION}" \
      org.opencontainers.image.revision="${KILN_REVISION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.base.digest="${BASE_DIGEST}"

COPY . /opt/kiln

# --no-deps: nothing Kiln imports is missing from the base (see above); --no-build-isolation: build with
# the base's own setuptools instead of fetching one; --no-cache-dir: no pip cache in the layer.
RUN test -n "${KILN_VERSION}" && test -n "${KILN_REVISION}" \
    || { echo "build with --build-arg KILN_VERSION=... --build-arg KILN_REVISION=..." >&2; exit 1; } \
 && pip install --no-cache-dir --no-deps --no-build-isolation /opt/kiln \
 && rm -rf /opt/kiln/build /opt/kiln/kiln.egg-info \
 && find /opt/kiln -name __pycache__ -type d -prune -exec rm -rf {} + \
 && KILN_VERSION="${KILN_VERSION}" python /opt/kiln/docker/check_image.py

ENV KILN_VERSION=${KILN_VERSION}
WORKDIR /opt/kiln
EXPOSE 8000
# The base's health check probes its own server on 8080; Kiln listens on --port (8000 by default) and
# answers GET /health there.
HEALTHCHECK NONE
ENTRYPOINT ["python", "-m", "kiln"]
CMD ["--help"]
