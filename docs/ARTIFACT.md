# Artifact Guide

This guide maps the public `structured_sparseGPT` repository to a reviewer-friendly artifact workflow for `StructPrune`. It is meant to make the release easier to inspect in the style of ICML, ICLR, NeurIPS, and similar artifact-review processes.

## What To Inspect First

- Root-level source files: compact implementation with the main entry points listed below.
- Root-level entry points: `llama_main.py`, `opt_main.py`.

## Environment Files

- No dependency manifest is tracked; follow the README install commands and imported packages in the main entry-point scripts.

## Minimal Verification

Run these checks in a fresh environment before launching expensive jobs:

```bash
python -m compileall -q .
```

If a smoke command is not tracked, use the README Quick Start with the smallest available seed, sample, or task count.

## Reproduction And Analysis Entry Points

These are the main tracked files to inspect for paper-scale or benchmark-scale reproduction. Some require arguments, credentials, downloaded benchmarks, or local data paths described in the README.

- `python llama_main.py`
- `python opt_main.py`

## Data, Credentials, And Generated Outputs

- Keep local dataset paths, downloaded corpora, checkpoints, and generated run artifacts outside git unless the README identifies them as small checked-in fixtures.
- Record dataset version, preprocessing command, seed, and hardware/runtime notes for every reproduced table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reviewer Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
