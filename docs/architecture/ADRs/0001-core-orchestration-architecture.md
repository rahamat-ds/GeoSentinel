# ADR 0001: Stage-Based Pipeline with a Shared Context Object

## Status
Accepted

## Context
GeoSentinel-AI must support fundamentally different task families
(classification, segmentation, detection, change detection) whose
experiment flows share almost no concrete steps, while still
guaranteeing that every run captures the same provenance (dataset
version, seed, environment, metrics, runtime, artifacts).

## Decision
Model an experiment as an ordered `Sequence[PipelineStage]` executed by
a `Pipeline`, communicating through one mutable `PipelineContext`
(artifacts + metrics dicts) rather than typed return values per stage.
`Pipeline.run()` owns seeding, timing, error containment, and
provenance finalization; stages own only their own logic.

## Alternatives Considered
1. **One big `run_experiment()` function per task type.** Rejected:
   duplicates provenance-capture logic across every task family and
   makes partial reuse (e.g. reusing benchmarking across tasks)
   impossible without copy-paste.
2. **Directed acyclic graph (DAG) of stages (Airflow/Kedro-style).**
   More expressive for branching/parallel pipelines, but adds
   scheduling complexity researchers don't need for a single-run
   experiment; revisit if multi-node/parallel stages become common.
3. **Typed stage-to-stage function signatures** (stage N's output type
   is stage N+1's input type). More static-typing rigor, but breaks
   down once stage count varies by task; the shared-context approach
   trades static typing for compositional flexibility, compensated by
   runtime `get_artifact(key, expected_type)` checks.

## Consequences
- New task families are added by writing new stages, not new engines.
- `context.artifacts: dict[str, Any]` is a typing weak point; mitigated
  by `PipelineContext.get_artifact` runtime type checks and by keeping
  the dict access pattern only inside stages, never in core.
- Full bitwise reproducibility is not promised across GPU
  driver/hardware changes -- only what's under software control
  (seeds, versions, environment) is captured. This must be stated
  explicitly in any published methods section referencing this
  platform.
- Single-process execution model for now; large-scale/distributed EO
  workloads will need a `Pipeline` variant (e.g. Ray/Dask-backed)
  implementing the same `PipelineStage` contract.