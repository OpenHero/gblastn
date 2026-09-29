# syntax=docker/dockerfile:1
# Published NVIDIA multi-platform indexes, verified 2026-09-29 (linux/arm64).
# JetPack 7.2.1 supplies CUDA 13.2.2; NVIDIA publishes these CUDA 13.2.1 images.
ARG CUDA_DEVEL_IMAGE=docker.io/nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:44a9504c6dfb50b1241464241b02a93871928f373de6f5a644cf5fe9f080aa63
ARG CUDA_RUNTIME_IMAGE=docker.io/nvidia/cuda:13.2.1-runtime-ubuntu24.04@sha256:4c17c2e7474ca1d470428b25e87e95756f51e7da3b663859894a03bc5a7ebdae
FROM ${CUDA_DEVEL_IMAGE} AS devel
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential gcc-13 g++-13 perl python3 zlib1g-dev libbz2-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

FROM devel AS build
ARG JOBS=4
ARG CUDA_ARCH=110
ARG CC=gcc-13
ARG CXX=g++-13
WORKDIR /src/gblastn
COPY c++ ./c++
COPY scripts/build.sh ./scripts/build.sh
COPY LICENSE.md ./LICENSE.md
RUN JOBS=${JOBS} CUDA_ARCH=${CUDA_ARCH} CC=${CC} CXX=${CXX} \
    BUILD_ROOT=/build/toolkit PREFIX=/opt/gblastn ./scripts/build.sh

# Includes nvcc, cuobjdump and Compute Sanitizer for target hardware validation.
FROM build AS validation
ENTRYPOINT []
COPY tests /src/gblastn/tests
COPY tests /opt/gblastn/tests
ENV PATH=/opt/gblastn/bin:${PATH}
WORKDIR /work
CMD ["blastn", "-version"]

FROM ${CUDA_RUNTIME_IMAGE} AS runtime
ENTRYPOINT []
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libstdc++6 libgomp1 zlib1g libbz2-1.0 python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/gblastn /opt/gblastn
COPY tests /opt/gblastn/tests
ENV PATH=/opt/gblastn/bin:${PATH}
RUN for bin in blastn makeblastdb blastdbcmd; do \
        ldd /opt/gblastn/bin/$bin >/tmp/ldd.txt && ! grep -q 'not found' /tmp/ldd.txt || exit 1; \
    done && rm /tmp/ldd.txt \
    && blastn -version && makeblastdb -version && blastdbcmd -version
WORKDIR /work
CMD ["blastn", "-help"]
