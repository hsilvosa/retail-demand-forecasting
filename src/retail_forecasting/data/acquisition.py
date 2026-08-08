"""Optional M5 acquisition through the official Kaggle CLI."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from typing import Any

from retail_forecasting.config import ProjectConfig
from retail_forecasting.data.quality import validate_source_files

COMPETITION = "m5-forecasting-accuracy"


def download_m5(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    existing = validate_source_files(config)
    if existing["valid"] and not force:
        return {"status": "already_available", "source": str(config.paths.source)}
    source = config.paths.source
    source.mkdir(parents=True, exist_ok=True)
    archive = source / f"{COMPETITION}.zip"
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(source),
        "--force",
    ]
    subprocess.run(command, check=True)
    if not archive.exists():
        candidates = list(source.glob("*.zip"))
        if len(candidates) != 1:
            raise FileNotFoundError("Kaggle download did not produce the expected archive")
        archive = candidates[0]
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(source)
    report = validate_source_files(config)
    if not report["valid"]:
        raise ValueError(f"Downloaded M5 files failed validation: {report}")
    archive.unlink(missing_ok=True)
    return {"status": "downloaded", "source": str(source), "validation": report}
