# Deadline-Aware Inference on a Shared GPU

Research on shared-GPU inference: measuring and controlling interference between latency-critical perception and vision-language model serving, with a focus on deadlines, freshness, and useful throughput.

## Current status

Early development: launch validation toward the first development pilot.

## Research book

The **Two Clocks research book** documents the experimental design, measurement methodology, and diagrams. Results and interpretation will be added with references to the underlying runs.

- [Book overview](wiki-research-book/index.qmd)
- [First shared-GPU pilot](wiki-research-book/experiments/r0a-pilot.qmd)
- [Measurement methodology](wiki-research-book/methods/measurement-methodology.qmd)

These links open the book’s source files in this repository.

## First research question

Can a fast vision workload and a bounded vision-language model run correctly on one GPU, and how does active sharing affect the fast service beyond the cost of keeping both deployments resident?

The planned pilot compares:

| Arm | Configuration |
| --- | --- |
| A | Fast vision workload alone; no slow server |
| B | Slow vision-language workload alone |
| A-resident | Fast vision active; the same slow deployment loaded but idle |
| C0 | Both workloads active, with a bounded queue |

The pilot will report both services, preserve failed, dropped, and unfinished work, and keep correctness, timeliness, and freshness distinct.

## Getting started

Follow the [CUDA development container instructions](containers/cuda/README.md) for host prerequisites and setup. Open the repository in VS Code and select **Dev Containers: Reopen in Container** using the CUDA configuration.

From the repository root inside the container, run:

```bash
bash scripts/smoke_test_env.sh
```

## Repository guide

| Path | Purpose |
| --- | --- |
| [.devcontainer/cuda/devcontainer.json](.devcontainer/cuda/devcontainer.json) | VS Code Dev Container configuration |
| [containers/cuda/](containers/cuda/) | CUDA Dockerfile and environment instructions |
| [requirements-dev.txt](requirements-dev.txt) | Pinned top-level Python dependencies |
| [scripts/smoke_test_env.sh](scripts/smoke_test_env.sh) | Small GPU tensor correctness check |
| [wiki-research-book/](wiki-research-book/) | Quarto research book: experiment design, methodology, diagrams, and results |