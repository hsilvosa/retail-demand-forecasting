# Local Operations

## Startup

Create `.env` from `.env.example`, start PostgreSQL, MinIO, MLflow, and Spark, then run preflight.
The full profile uses the `pipeline-gpu` service and stops immediately when CUDA is unavailable.
The default development services are CPU-only and do not download CUDA dependencies.

## Recovery

Use `pipeline status` to inspect stage manifests. A pending stage can be rerun from that point with
`--from`. Do not use `--force` for ordinary recovery; it intentionally invalidates fingerprint
reuse and rewrites downstream Delta tables.

MLflow artifacts remain in the MinIO volume and metadata remains in PostgreSQL. Removing either
volume destroys part of the experiment record. Generated lakehouse and service volumes are not
Git-controlled backups.

## Common failures

If Spark cannot download Delta coordinates, rebuild the image while network access is available.
If MLflow cannot upload an artifact, verify the MinIO health check, bucket initialization, endpoint
URL, and credentials. If GPU training falls back or fails, run `nvidia-smi` inside the pipeline
GPU container and verify Docker Desktop GPU integration.

## Reproducibility record

Every MLflow parent run stores the resolved configuration, Git SHA, profile, nested fold runs,
metrics, tuning parameters, model artifact, and explanations. Delta output rows carry the pipeline
run ID. Source checksums connect Bronze data to the original CSV files.
