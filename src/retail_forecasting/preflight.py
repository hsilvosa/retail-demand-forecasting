"""Environment checks that fail before an expensive pipeline starts."""

from __future__ import annotations

import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

from retail_forecasting.config import ProjectConfig


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def run_preflight(config: ProjectConfig, check_ports: bool = False) -> list[CheckResult]:
    source = config.paths.source
    missing = [name for name in config.data.required_files if not (source / name).exists()]
    results = [
        CheckResult("source_data", not missing, "ok" if not missing else f"missing: {missing}"),
        CheckResult("java", shutil.which("java") is not None, "required by Spark"),
        CheckResult("gpu", _gpu_available(), "required for full; optional for dev"),
    ]
    if check_ports:
        for port in (5000, 8080, 8501, 8888, 9000, 9001):
            results.append(CheckResult(f"port_{port}", _port_available(port), "must be free"))
    if config.models.require_gpu and not next(item.ok for item in results if item.name == "gpu"):
        results.append(CheckResult("profile_gpu_requirement", False, "full profile requires CUDA"))
    return results


def assert_preflight(config: ProjectConfig) -> None:
    failed = [result for result in run_preflight(config) if not result.ok]
    blocking = [r for r in failed if r.name != "gpu" or config.models.require_gpu]
    if blocking:
        details = "; ".join(f"{item.name}: {item.detail}" for item in blocking)
        raise RuntimeError(f"Preflight failed: {details}")
