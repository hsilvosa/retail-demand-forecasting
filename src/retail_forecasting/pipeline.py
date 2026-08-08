"""Small dependency-aware pipeline runner used by CLI and notebooks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from retail_forecasting.config import ProjectConfig

log = structlog.get_logger()
StageFunction = Callable[[ProjectConfig, str], dict[str, Any]]


@dataclass(frozen=True)
class Stage:
    name: str
    function: StageFunction


def _fingerprint(config: ProjectConfig, stage: str) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True) + stage
    return hashlib.sha256(payload.encode()).hexdigest()


def _manifest_path(config: ProjectConfig, stage: str) -> Path:
    return config.paths.state / config.profile / f"{stage}.json"


def _is_current(config: ProjectConfig, stage: str) -> bool:
    path = _manifest_path(config, stage)
    if not path.exists():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest.get("fingerprint") == _fingerprint(config, stage)


def _write_manifest(
    config: ProjectConfig, stage: str, run_id: str, outputs: dict[str, Any]
) -> None:
    path = _manifest_path(config, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "profile": config.profile,
                "run_id": run_id,
                "completed_at": datetime.now(UTC).isoformat(),
                "fingerprint": _fingerprint(config, stage),
                "outputs": outputs,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def default_stages() -> list[Stage]:
    from retail_forecasting.data.bronze import run_bronze
    from retail_forecasting.data.gold import run_gold
    from retail_forecasting.data.silver import run_silver
    from retail_forecasting.forecasting.workflow import run_forecasting
    from retail_forecasting.inventory import run_inventory

    return [
        Stage("bronze", run_bronze),
        Stage("silver", run_silver),
        Stage("gold", run_gold),
        Stage("forecasting", run_forecasting),
        Stage("inventory", run_inventory),
    ]


def select_stages(
    stages: list[Stage], from_stage: str | None = None, to_stage: str | None = None
) -> list[Stage]:
    names = [stage.name for stage in stages]
    start = names.index(from_stage) if from_stage else 0
    end = names.index(to_stage) + 1 if to_stage else len(stages)
    if start >= end:
        raise ValueError("from_stage must precede or equal to_stage")
    return stages[start:end]


def run_pipeline(
    config: ProjectConfig,
    run_id: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
    force: bool = False,
) -> list[str]:
    completed: list[str] = []
    for stage in select_stages(default_stages(), from_stage, to_stage):
        if not force and _is_current(config, stage.name):
            log.info("stage_skipped", stage=stage.name, reason="fingerprint_match")
            continue
        log.info("stage_started", stage=stage.name, profile=config.profile, run_id=run_id)
        outputs = stage.function(config, run_id)
        _write_manifest(config, stage.name, run_id, outputs)
        completed.append(stage.name)
        log.info("stage_completed", stage=stage.name, outputs=outputs)
    return completed


def pipeline_status(config: ProjectConfig) -> list[dict[str, str]]:
    status = []
    for stage in default_stages():
        path = _manifest_path(config, stage.name)
        state = "current" if _is_current(config, stage.name) else "pending"
        status.append({"stage": stage.name, "status": state, "manifest": str(path)})
    return status
