# Artifact Guide

Operational notes for reproducing `StructPrune` from the public `structured_sparseGPT` repository.

## Review Path

- Root-level source files: compact implementation with the main entry points listed below.
- Root-level entry points: `llama_main.py`, `opt_main.py`.

## Environment Files

- No dependency manifest is tracked; follow the README install commands and imported packages in the main entry-point scripts.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
```

If no smoke command is tracked, use the README Quick Start with the smallest seed, sample, or task count.

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

- `python llama_main.py`
- `python opt_main.py`

## Data And Outputs

- Keep local dataset paths, downloaded corpora, checkpoints, and generated run artifacts outside git unless the README identifies them as small checked-in fixtures.
- Record dataset version, preprocessing command, seed, and hardware/runtime notes for every reproduced table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
