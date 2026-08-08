# Architecture

## Design goals

The project must make the transition from a local portfolio repository to a managed Databricks
workspace understandable. Storage, computation, orchestration, experiment tracking, and serving
are separated even though they run on one machine.

## Runtime services

The pipeline image contains Python 3.11, Java 17, PySpark 3.5, Delta Lake 3.3, MLflow, the model
libraries, JupyterLab, and Streamlit. Spark master and worker processes use that image so the
driver and executors have identical dependencies. The pipeline and Jupyter services request the
NVIDIA device.

PostgreSQL owns MLflow metadata. MinIO owns artifacts through an S3-compatible endpoint. This
separation exercises the same client and credential boundaries used with managed stores without
requiring a cloud account.

## Medallion layers

Bronze preserves source structure and records ingestion lineage. Silver owns typing,
normalization, reference joins, and data-quality gates. Gold owns model-ready features,
hierarchies, forecasts, metrics, explanations, and inventory outputs.

Delta table paths are part of the internal implementation. Consumers use stable table names and
contracts rather than inspecting transaction log files. Every mutating pipeline stage writes its
own completion manifest only after its output succeeds.

## Orchestration

The Python DAG orders Bronze, Silver, Gold, forecasting, and inventory. A stage fingerprint is
derived from the resolved profile and stage name. This gives deterministic restart behavior while
keeping the orchestration layer small enough to read and test.

The equivalent Databricks deployment would map these stages to tasks in a multi-task Job, move
paths to catalog tables, replace MinIO with object storage, and point the same MLflow calls to the
workspace tracking service.
