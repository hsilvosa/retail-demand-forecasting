"""Command line interface for local and containerized workflows."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import typer

from retail_forecasting.config import load_config
from retail_forecasting.logging import configure_logging

app = typer.Typer(no_args_is_help=True, help="Retail demand forecasting platform")
pipeline_app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
forecast_app = typer.Typer(no_args_is_help=True)
inventory_app = typer.Typer(no_args_is_help=True)
notebook_app = typer.Typer(no_args_is_help=True)
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(forecast_app, name="forecast")
app.add_typer(inventory_app, name="inventory")
app.add_typer(notebook_app, name="notebooks")

ProfileOption = Annotated[str, typer.Option(help="Configuration profile: dev, full, or test")]


@app.callback()
def root(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    configure_logging(verbose)


@app.command("preflight")
def preflight(profile: ProfileOption = "dev", check_ports: bool = False) -> None:
    from retail_forecasting.preflight import run_preflight

    config = load_config(profile)
    results = run_preflight(config, check_ports=check_ports)
    for result in results:
        typer.echo(f"{'PASS' if result.ok else 'FAIL'} {result.name}: {result.detail}")
    blocking = [r for r in results if not r.ok and (r.name != "gpu" or config.models.require_gpu)]
    if blocking:
        raise typer.Exit(1)


@pipeline_app.command("run")
def pipeline_run(
    profile: ProfileOption = "dev",
    from_stage: Annotated[str | None, typer.Option("--from")] = None,
    to_stage: Annotated[str | None, typer.Option("--to")] = None,
    force: bool = False,
    run_id: str | None = None,
) -> None:
    from retail_forecasting.pipeline import run_pipeline
    from retail_forecasting.preflight import assert_preflight

    config = load_config(profile)
    assert_preflight(config)
    completed = run_pipeline(
        config, run_id or uuid.uuid4().hex[:12], from_stage, to_stage, force
    )
    typer.echo(json.dumps({"completed": completed, "profile": profile}))


@pipeline_app.command("status")
def status(profile: ProfileOption = "dev") -> None:
    from retail_forecasting.pipeline import pipeline_status

    typer.echo(json.dumps(pipeline_status(load_config(profile)), indent=2))


@data_app.command("validate")
def data_validate(profile: ProfileOption = "dev") -> None:
    from retail_forecasting.data.quality import validate_source_files

    report = validate_source_files(load_config(profile))
    typer.echo(json.dumps(report, indent=2))
    if not report["valid"]:
        raise typer.Exit(1)


@data_app.command("download")
def data_download(profile: ProfileOption = "dev", force: bool = False) -> None:
    from retail_forecasting.data.acquisition import download_m5

    result = download_m5(load_config(profile), force=force)
    typer.echo(json.dumps(result, indent=2))


@model_app.command("backtest")
def model_backtest(profile: ProfileOption = "dev", run_id: str | None = None) -> None:
    from retail_forecasting.forecasting.workflow import run_forecasting

    result = run_forecasting(load_config(profile), run_id or uuid.uuid4().hex[:12])
    typer.echo(json.dumps(result, indent=2))


@model_app.command("train")
def model_train(profile: ProfileOption = "dev", run_id: str | None = None) -> None:
    model_backtest(profile, run_id)


@model_app.command("promote")
def model_promote(profile: ProfileOption = "dev", run_id: str = "") -> None:
    from retail_forecasting.tracking import promote_candidate

    result = promote_candidate(load_config(profile), run_id or None)
    typer.echo(json.dumps(result, indent=2))


@forecast_app.command("run")
def forecast_run(profile: ProfileOption = "dev", run_id: str | None = None) -> None:
    from retail_forecasting.forecasting.workflow import run_final_forecast

    result = run_final_forecast(load_config(profile), run_id or uuid.uuid4().hex[:12])
    typer.echo(json.dumps(result, indent=2))


@inventory_app.command("simulate")
def inventory_simulate(profile: ProfileOption = "dev", run_id: str | None = None) -> None:
    from retail_forecasting.inventory import run_inventory

    result = run_inventory(load_config(profile), run_id or uuid.uuid4().hex[:12])
    typer.echo(json.dumps(result, indent=2))


@notebook_app.command("run")
def notebooks_run(
    profile: ProfileOption = "dev",
    notebook: Annotated[str | None, typer.Option(help="Notebook stem or all")] = "all",
) -> None:
    from retail_forecasting.notebooks import execute_notebooks

    outputs = execute_notebooks(profile, notebook or "all", Path("notebooks"))
    typer.echo(json.dumps([str(path) for path in outputs], indent=2))
