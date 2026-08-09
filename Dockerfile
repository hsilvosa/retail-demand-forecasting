# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS base

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYSPARK_PYTHON=/usr/local/bin/python \
    PYSPARK_DRIVER_PYTHON=/usr/local/bin/python \
    SPARK_NO_DAEMONIZE=true

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl git libgomp1 openjdk-17-jre-headless tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install .

COPY config ./config
COPY sql ./sql
COPY scripts ./scripts

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["retail-forecast", "--help"]

FROM base AS spark
RUN --mount=type=cache,target=/root/.cache/pip pip install '.[lakehouse]'

FROM spark AS pipeline-base
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install '.[tracking,trees]'

FROM pipeline-base AS pipeline

FROM pipeline-base AS pipeline-gpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.5.1' && \
    pip install '.[deep]'

FROM pipeline AS jupyter
RUN --mount=type=cache,target=/root/.cache/pip pip install '.[notebooks]'
COPY notebooks ./notebooks

FROM pipeline-gpu AS jupyter-gpu
RUN --mount=type=cache,target=/root/.cache/pip pip install '.[notebooks]'
COPY notebooks ./notebooks

FROM base AS mlflow
RUN --mount=type=cache,target=/root/.cache/pip pip install '.[tracking]'

FROM base AS dashboard
RUN --mount=type=cache,target=/root/.cache/pip pip install '.[app]'
COPY app ./app
