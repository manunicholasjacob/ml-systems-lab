# The agent image: somewhere a run can happen that is not a host.
#
# Two rules govern everything below.
#
# 1. Pin by digest, never by tag. A tag is a moving target, and two runs a week apart
#    against the same tag are not the same experiment. The base is pinned here; the
#    digest of the built image is recorded in every run record by K8sDevice.image_id().
#
# 2. The package is NOT baked in. It is streamed into the container at run time, exactly
#    as the SSH device streams it to the Pi. That keeps one image valid across every
#    version of the harness, and it means a code change does not require a rebuild.
#
# Build:
#   docker build -t mlsyslab-agent:0.2.0 -f docker/Dockerfile .
#
# Load into a k3s node (no registry needed):
#   docker save mlsyslab-agent:0.2.0 | sudo k3s ctr images import -
#
# See docker/README.md for the full sequence, including how to extract the llama.cpp
# binaries so the host arm of the containerization experiment runs the same bytes.

# ------------------------------------------------------------------ build llama.cpp
FROM debian:bookworm-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171 AS build

ARG LLAMACPP_COMMIT=d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b

# GGML_NATIVE is ON by default because this image exists to measure a specific machine,
# and a generic build would measure a compiler decision instead. The consequence is that
# the image is portable only across CPUs with the same feature set, which is the correct
# trade for a benchmark. Pass --build-arg GGML_NATIVE=OFF for a portable image.
ARG GGML_NATIVE=ON

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp /src \
    && cd /src \
    && git checkout ${LLAMACPP_COMMIT} \
    && git rev-parse HEAD > /llamacpp-commit.txt

RUN cmake -S /src -B /src/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=${GGML_NATIVE} \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
    && cmake --build /src/build --config Release -j "$(nproc)" \
        --target llama-bench llama-server llama-cli

# Collect the binaries and the shared objects they were linked against, so the runtime
# stage does not have to guess which ggml backend libraries got built.
RUN mkdir -p /out/bin /out/lib \
    && cp /src/build/bin/llama-bench /src/build/bin/llama-server /src/build/bin/llama-cli /out/bin/ \
    && find /src/build/bin -name 'lib*.so*' -exec cp -a {} /out/lib/ \; \
    && cp /llamacpp-commit.txt /out/

# ---------------------------------------------------------------------- runtime
FROM python:3.11-slim-bookworm@sha256:0bee7276f83efd4a1ee05bbbf4281d95ed28e079220a9457f25a93e3f1e3c31b

# libgomp1 is llama.cpp's OpenMP runtime, libcurl4 is what LLAMA_CURL linked against, and
# procps gives the agent a way to see what else is running when a number looks wrong.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libcurl4 procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/bin/ /usr/local/bin/
COPY --from=build /out/lib/ /usr/local/lib/mlsyslab/
COPY --from=build /out/llamacpp-commit.txt /etc/mlsyslab-llamacpp-commit

ENV LD_LIBRARY_PATH=/usr/local/lib/mlsyslab \
    MLSYSLAB_WORKDIR=/opt/mlsyslab \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The harness unpacks itself here and sets PYTHONPATH to match. Writable by any uid so
# the pod can run as a non-root user without a fight over ownership.
RUN mkdir -p /opt/mlsyslab && chmod 1777 /opt/mlsyslab
WORKDIR /opt/mlsyslab

# Proof at build time that the three things the harness needs are actually present, so a
# broken image fails here rather than three hours into a sweep.
RUN python3 -c "import json, sys; print(sys.version)" \
    && tar --version > /dev/null \
    && llama-bench --help > /dev/null 2>&1 || true

LABEL org.opencontainers.image.title="mlsyslab-agent" \
      org.opencontainers.image.description="Run target for ml-systems-lab: python3, tar and llama.cpp" \
      org.opencontainers.image.source="https://github.com/manunicholasjacob/ml-systems-lab" \
      org.opencontainers.image.licenses="MIT"

CMD ["sh", "-c", "trap 'exit 0' TERM; while true; do sleep 3600 & wait $!; done"]
