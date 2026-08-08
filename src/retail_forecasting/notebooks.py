"""Papermill execution for the editable notebook workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def execute_notebooks(profile: str, notebook: str, notebook_dir: Path) -> list[Path]:
    import papermill as pm

    sources = sorted(notebook_dir.glob("*.ipynb"))
    if notebook != "all":
        sources = [path for path in sources if path.stem == notebook]
    if not sources:
        raise FileNotFoundError(f"No notebooks matched {notebook!r}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("artifacts/notebooks") / profile / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    run_id = f"notebook-{stamp.lower()}"
    for source in sources:
        destination = output_dir / source.name
        pm.execute_notebook(
            source,
            destination,
            parameters={"profile": profile, "run_id": run_id, "force": False},
            kernel_name="python3",
        )
        outputs.append(destination)
    return outputs
