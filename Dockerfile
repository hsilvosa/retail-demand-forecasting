FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYSPARK_PYTHON=/opt/conda/bin/python \
    PYSPARK_DRIVER_PYTHON=/opt/conda/bin/python \
    SPARK_NO_DAEMONIZE=true

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl git openjdk-17-jre-headless tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir '.[all]'

COPY config ./config
COPY notebooks ./notebooks
COPY app ./app
COPY scripts ./scripts

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["retail-forecast", "--help"]
