FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYSPARK_PYTHON=/opt/conda/bin/python \
    PYSPARK_DRIVER_PYTHON=/opt/conda/bin/python \
    SPARK_NO_DAEMONIZE=true

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake curl git libboost-dev libboost-filesystem-dev \
      libboost-system-dev ninja-build openjdk-17-jre-headless tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir '.[lakehouse,modeling,app,notebooks,dev]' && \
    pip uninstall -y lightgbm && \
    CMAKE_ARGS="-DUSE_CUDA=ON" pip install --no-cache-dir --no-binary lightgbm 'lightgbm>=4.5,<5'

COPY config ./config
COPY sql ./sql
COPY notebooks ./notebooks
COPY app ./app
COPY scripts ./scripts

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["retail-forecast", "--help"]
