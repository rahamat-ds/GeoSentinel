"""Command-line entry point: `geosentinel run --config <path>`."""
from __future__ import annotations

from pathlib import Path

import typer

from geosentinel.core.configuration.loader import ConfigLoader
from geosentinel.core.logging.setup import configure_logging
from geosentinel.core.plugins.manager import PluginManager

app = typer.Typer(help="GeoSentinel-AI: Trustworthy Environmental Intelligence, orchestrated.")


@app.command()
def run(config: Path = typer.Option(..., "--config", "-c", help="Path to an experiment YAML")) -> None:
    """Run an experiment end-to-end from a validated config file."""
    configure_logging()
    experiment_config = ConfigLoader.load(config)

    manager = PluginManager()
    activated = manager.activate_all()
    typer.echo(f"Activated plugins: {activated or 'none'}")
    typer.echo(
        f"Loaded experiment '{experiment_config.experiment_name}' "
        f"(task={experiment_config.model.task_type.value}). "
        f"Wire this ExperimentConfig into a Pipeline of PipelineStages to execute it."
    )


@app.command("list-plugins")
def list_plugins() -> None:
    """List installed GeoSentinel plugins without activating them."""
    for ep in PluginManager().discover():
        typer.echo(f"{ep.name} -> {ep.value}")


if __name__ == "__main__":
    app()