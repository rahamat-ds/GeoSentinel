"""Generates a publication-ready Markdown report + LaTeX table + confusion-matrix figure.

Kept template-free (plain f-strings): a templating engine (Jinja2) is
a reasonable future addition, but for the report structure mandated by
the Research Standards, an explicit function is easier to audit
line-by-line than a template file.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from geosentinel.benchmarking.metrics import BenchmarkResult
from geosentinel.core.provenance.capture import ExperimentRecord


def _plot_confusion_matrix(result: BenchmarkResult, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(result.confusion, cmap="Blues")
    ax.set_xticks(range(len(result.class_labels)))
    ax.set_yticks(range(len(result.class_labels)))
    ax.set_xticklabels(result.class_labels, rotation=45, ha="right")
    ax.set_yticklabels(result.class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(result.confusion.shape[0]):
        for j in range(result.confusion.shape[1]):
            ax.text(j, i, str(result.confusion[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


class ReportGenerator:
    """Assembles record + benchmark results into reproducible report artifacts."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self, record: ExperimentRecord, benchmark: BenchmarkResult, formats: list[str], generate_figures: bool = True
    ) -> dict[str, Path]:
        outputs: dict[str, Path] = {}
        if generate_figures:
            outputs["confusion_matrix_figure"] = _plot_confusion_matrix(
                benchmark, self._output_dir / "confusion_matrix.png"
            )
        if "markdown" in formats:
            outputs["markdown"] = self._write_markdown(record, benchmark)
        if "latex" in formats:
            outputs["latex"] = self._write_latex(record, benchmark)
        return outputs

    def _write_markdown(self, record: ExperimentRecord, benchmark: BenchmarkResult) -> Path:
        path = self._output_dir / "report.md"
        stages = chr(10).join(f"- {s}" for s in record.stage_log)
        content = f"""# Experiment Report: {record.experiment_name}

**Experiment ID:** `{record.experiment_id}`
**Status:** {record.status}
**Seed:** {record.seed}
**Runtime:** {record.runtime_seconds}s

## Software Environment
- Python: {record.software_environment.get('python_version')}
- Platform: {record.software_environment.get('platform')}
- Git commit: {record.software_environment.get('git_commit')}

## Hardware Environment
- Processor: {record.hardware_environment.get('processor')}
- GPU: {record.hardware_environment.get('gpu')}

## Results

{benchmark.to_markdown_table()}

![Confusion Matrix](confusion_matrix.png)

## Pipeline Stages Executed
{stages}
"""
        path.write_text(content, encoding="utf-8")
        return path

    def _write_latex(self, record: ExperimentRecord, benchmark: BenchmarkResult) -> Path:
        path = self._output_dir / "results_table.tex"
        path.write_text(benchmark.to_latex_table(caption=f"Results for {record.experiment_name}"), encoding="utf-8")
        return path