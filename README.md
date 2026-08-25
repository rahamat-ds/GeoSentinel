# GeoSentinel

### Trustworthy Environmental Intelligence for Reproducible Research

GeoSentinel is an open research platform for building, executing, evaluating, and documenting trustworthy environmental intelligence experiments.
It is designed around a simple idea:

> **Environmental AI should be reproducible, auditable, explainable, and uncertainty-aware—not merely accurate.**

GeoSentinel provides the orchestration and research infrastructure needed to turn an environmental ML experiment into a structured, traceable research run.
The platform is being developed with **Earth Observation and environmental monitoring** as its primary domains, with future support for applications such as land-cover mapping, ecological monitoring, biodiversity assessment, climate analysis, and marine biological monitoring.

---

## Why GeoSentinel?

Environmental machine-learning research often involves a fragmented workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
Model
   ↓
Inference
   ↓
Metrics
   ↓
Explainability
   ↓
Uncertainty
   ↓
Report
```

Each step may live in a different notebook, script, or experiment directory.

This creates a problem.

Months later, it can become difficult to answer:

* Which exact dataset version was used?
* Which configuration produced this result?
* What random seed was used?
* Which model and checkpoint were evaluated?
* What preprocessing was applied?
* Which metrics were calculated?
* How long did each stage take?
* Can another researcher reproduce the experiment?
* How confident should we be in the model's predictions?

GeoSentinel treats these concerns as **first-class components of the research workflow**.

---

## Core Philosophy

GeoSentinel is not intended to compete with PyTorch, TorchGeo, scikit-learn, Rasterio, or other specialized scientific libraries.

Instead, it sits **above them** as an orchestration and reproducibility layer.

```text
                ┌──────────────────────────┐
                │      Research Question   │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │      Experiment Config   │
                └────────────┬─────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────┐
        │            GeoSentinel Engine            │
        │                                          │
        │  Dataset → Preprocess → Model → Evaluate │
        │             ↓                            │
        │      Explainability / Uncertainty        │
        │             ↓                            │
        │       Provenance / Reporting             │
        └────────────────────┬─────────────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ Reproducible Experiment  │
                └──────────────────────────┘
```

The goal is to make trustworthy experimentation part of the architecture rather than something researchers have to remember manually.

---

# Current Status

🚧 **Early research prototype — active development**

The current release focuses on the **core orchestration engine**.

Implemented:

* Typed experiment configuration with Pydantic
* YAML configuration loading and validation
* Environment-variable interpolation
* Configuration inheritance
* Stage-based pipeline execution
* Pipeline context and artifact passing
* Stage precondition checking
* Experiment identifiers
* Global random-seed management
* Experiment provenance capture
* Structured logging
* Generic component registry
* Plugin architecture foundation
* CLI foundation
* Benchmarking interfaces
* Explainability interfaces
* Uncertainty interfaces
* Reporting interfaces
* Automated pipeline tests
* Ruff-based code quality checks

The current implementation deliberately uses lightweight/fake stages in its tests. Real environmental datasets, models, and domain plugins are being added incrementally.

---

# Architecture

GeoSentinel is organized around a modular research pipeline.

```text
geosentinel/
│
├── core/
│   ├── configuration/
│   │   ├── schema.py
│   │   └── loader.py
│   │
│   ├── pipeline/
│   │   ├── context.py
│   │   ├── pipeline.py
│   │   └── stage.py
│   │
│   ├── plugins/
│   │   ├── base.py
│   │   └── manager.py
│   │
│   ├── provenance/
│   │   ├── capture.py
│   │   └── seed.py
│   │
│   ├── registry/
│   │   └── registry.py
│   │
│   └── logging/
│       └── setup.py
│
├── benchmarking/
├── explainability/
├── uncertainty/
├── reporting/
└── cli/
```

The architecture separates **research orchestration** from **domain-specific implementations**.

This means the same engine can eventually support very different experiments:

```text
Earth Observation
        │
        ├── Land-cover classification
        ├── Semantic segmentation
        ├── Change detection
        ├── Flood mapping
        └── Wildfire monitoring

Marine Monitoring
        │
        ├── Species detection
        ├── Habitat classification
        └── Population monitoring

Biodiversity
        │
        ├── Species distribution
        ├── Acoustic monitoring
        └── Camera-trap analysis
```

without rebuilding the underlying experiment infrastructure each time.

---

# The Pipeline

A GeoSentinel experiment is represented as an ordered collection of stages.

Conceptually:

```text
┌──────────────┐
│ Load Dataset │
└──────┬───────┘
       ↓
┌──────────────┐
│ Preprocessing│
└──────┬───────┘
       ↓
┌──────────────┐
│   Inference  │
└──────┬───────┘
       ↓
┌──────────────┐
│ Explainability│
└──────┬───────┘
       ↓
┌──────────────┐
│  Uncertainty │
└──────┬───────┘
       ↓
┌──────────────┐
│  Benchmarking│
└──────┬───────┘
       ↓
┌──────────────┐
│   Reporting  │
└──────────────┘
```

Each stage has explicit inputs and outputs.

A stage can declare preconditions such as:

```text
requires:
    predictions
    ground_truth
```

If those artifacts do not exist, the pipeline fails early with a meaningful error rather than producing an obscure downstream exception.

---

# Configuration

Experiments are defined through validated YAML configuration.

A simplified example:

```yaml
experiment_name: land_cover_baseline

seed: 42

dataset:
  name: sentinel2_landcover
  version: "2024.03"
  root: ./datasets/satellite/sentinel2_landcover
  split: test

model:
  name: resnet50_landcover
  architecture: resnet50
  task_type: classification
  num_classes: 10

explainability:
  enabled: true
  method: vanilla_gradient

uncertainty:
  enabled: true
  method: mc_dropout

benchmarking:
  metrics:
    - accuracy
    - f1_macro
```

The configuration is validated before the experiment begins.

Scientific parameters that affect reproducibility, such as the random seed and dataset version, are intentionally treated as explicit configuration rather than hidden defaults.

---

# Provenance

A central objective of GeoSentinel is to make experiment provenance automatic.

A completed experiment will produce an experiment record containing information such as:

```text
Experiment ID
Configuration
Dataset information
Model information
Random seed
Environment information
Stage execution
Stage durations
Metrics
Experiment status
Output locations
```

The intention is that a researcher should not need to manually maintain a notebook saying:

> "I think I used version X with seed 42 and probably this checkpoint."

The experiment itself should record what happened.

---

# Extensibility

GeoSentinel uses registries and a plugin architecture to avoid hard-coding every model or environmental domain into the core.

Conceptually:

```python
@MODEL_REGISTRY.register("my_model")
class MyModel:
    ...
```

The same mechanism can support:

* Models
* Dataset providers
* Explainability methods
* Uncertainty estimators
* Benchmarks
* Domain plugins

This allows future components to evolve independently from the core engine.

---

# Trustworthy AI

GeoSentinel is being developed around three complementary questions.

### 1. What did the model predict?

Standard predictive performance:

* Accuracy
* Precision
* Recall
* F1
* IoU
* Task-specific metrics

### 2. Why did the model make the prediction?

Potential methods include:

* Saliency
* Integrated Gradients
* Grad-CAM
* SHAP
* Other domain-appropriate explanation methods

### 3. How certain is the model?

Potential methods include:

* Monte Carlo Dropout
* Deep Ensembles
* Conformal Prediction
* Calibration analysis
* Predictive uncertainty maps

The goal is not simply to produce a prediction.

It is to produce a prediction accompanied by enough information to evaluate whether that prediction should be trusted.

---

# Research Direction

The initial scientific focus is:

## AI for Earth Observation

Potential applications include:

* Land-cover classification
* Semantic segmentation
* Remote-sensing change detection
* Flood and wildfire mapping
* Environmental monitoring
* Ecosystem assessment

A second major direction is:

## AI-enabled Marine Biological Monitoring

Potential applications include:

* Species detection
* Marine habitat classification
* Biodiversity monitoring
* Underwater imagery analysis
* Population and ecological monitoring

The long-term objective is to provide a common research infrastructure across these domains while keeping domain-specific components modular.

---

# Technology

### Core

* Python
* Pydantic
* PyYAML
* NumPy
* Pandas
* scikit-learn
* Structlog
* Typer
* Matplotlib

### Development

* uv
* pytest
* pytest-cov
* Ruff
* mypy
* Git

### Planned / Optional Scientific Stack

* PyTorch
* TorchGeo
* Rasterio
* GeoPandas
* GDAL
* MLflow
* Captum
* SHAP

Heavy dependencies are intentionally optional so that the core engine remains lightweight and testable without a GPU.

---

# Installation

Clone the repository:

```bash
git clone git@github.com:<your-username>/GeoSentinel.git
cd GeoSentinel
```

Create the environment and install the development dependencies:

```bash
uv sync --extra dev
```

Run the test suite:

```bash
uv run python -m pytest -v
```

Run the CLI:

```bash
uv run geosentinel --help
```

Run code-quality checks:

```bash
uv run ruff check geosentinel tests
```

---

# Current Example

The repository currently contains an example experiment configuration:

```text
configs/
└── experiments/
    └── example_land_cover_classification.yaml
```

It represents the intended structure of a future Earth Observation experiment.

The dataset and model components required to execute a real Sentinel-2 experiment are still under development.

---

# Development Roadmap

### Phase 1 — Research Engine

* [x] Configuration system
* [x] Pipeline engine
* [x] Pipeline context
* [x] Stage contracts
* [x] Provenance foundation
* [x] Registry system
* [x] Plugin architecture
* [x] CLI foundation
* [x] Core pipeline tests

### Phase 2 — Real Experiments

* [ ] Dataset provider abstraction
* [ ] Local raster dataset support
* [ ] Earth Observation dataset integration
* [ ] Model provider abstraction
* [ ] Baseline PyTorch model
* [ ] Real inference pipeline
* [ ] Real benchmarking pipeline
* [ ] End-to-end Earth Observation experiment

### Phase 3 — Trustworthy Environmental AI

* [ ] Explainability pipeline
* [ ] Uncertainty estimation
* [ ] Calibration analysis
* [ ] Uncertainty visualization
* [ ] Explanation benchmarking
* [ ] Reproducible experiment reports

### Phase 4 — Research Platform

* [ ] MLflow integration
* [ ] Experiment comparison
* [ ] Dataset registry
* [ ] Model registry
* [ ] Domain plugin system
* [ ] Experiment dashboard
* [ ] Publication-ready reporting

### Phase 5 — Environmental Intelligence

* [ ] Earth Observation domain plugin
* [ ] Marine monitoring domain plugin
* [ ] Biodiversity applications
* [ ] Large-scale processing
* [ ] Distributed execution
* [ ] Research benchmarks
* [ ] Collaborative scientific workflows

---

# Research Goals

GeoSentinel is ultimately intended to investigate a broader question:

> **Can environmental AI systems be made not only accurate, but reproducible, interpretable, uncertainty-aware, and scientifically auditable by design?**

The project therefore treats software engineering and scientific methodology as equally important.

A model that achieves 95% accuracy but cannot explain its data provenance, quantify its uncertainty, or reproduce its results is not necessarily a reliable scientific instrument.

GeoSentinel aims to provide the infrastructure needed to move toward that standard.

---

# Project Status

**Status:** Active research / early-stage open-source development

This project is currently being developed as an independent research and engineering project.

The architecture is intentionally evolving. APIs, configuration schemas, plugin contracts, and directory structures may change as real environmental workloads are introduced.

---

# Contributing

Contributions, research ideas, experiments, datasets, and domain expertise are welcome.

Before contributing a major feature, please consider whether it:

1. Improves reproducibility.
2. Improves scientific validity.
3. Improves extensibility.
4. Adds meaningful environmental capability.
5. Can be tested and documented.

The goal is not to maximize the number of features.

The goal is to build useful scientific infrastructure.

---

# License

GeoSentinel is released under the **Apache License 2.0**.

---

## Vision

GeoSentinel started as an experiment in building environmental AI software.

The larger ambition is to build something closer to a **research operating system for environmental intelligence**:

```text
Research Question
       ↓
Data
       ↓
Experiment
       ↓
Model
       ↓
Prediction
       ↓
Explanation
       ↓
Uncertainty
       ↓
Evaluation
       ↓
Provenance
       ↓
Reproducible Evidence
```

**Build the experiment. Record the evidence. Trust the result.**
