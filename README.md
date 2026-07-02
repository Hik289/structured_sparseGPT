<div align="center">

# StructPrune

### Structured Global Pruning Asymptotics with `O(sqrt(N))` GPU Memory

Official implementation for the arXiv paper:

**StructPrune: Structured Global Pruning asymptotics with `O(sqrt(N))` GPU Memory**

<p>
  <a href="https://arxiv.org/abs/2510.03246"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2510.03246-b31b1b.svg"></a>
  <a href="https://arxiv.org/pdf/2510.03246"><img alt="Paper PDF" src="https://img.shields.io/badge/Paper-PDF-1f6feb.svg"></a>
  <a href="#quickstart"><img alt="Quickstart" src="https://img.shields.io/badge/Quickstart-OPT%20%7C%20LLaMA-2ea44f.svg"></a>
  <a href="#citation"><img alt="Citation" src="https://img.shields.io/badge/Cite-BibTeX-8a63d2.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-34d058.svg"></a>
</p>

<p>
  <a href="#overview">Overview</a> |
  <a href="#installation">Installation</a> |
  <a href="#quickstart">Quickstart</a> |
  <a href="#usage">Usage</a> |
  <a href="#citation">Citation</a>
</p>

</div>

## Overview

StructPrune studies a practical bottleneck in large language model pruning:
global structured pruning can preserve model quality better than purely local
layer-wise pruning, but naively solving the global problem requires memory that
scales with the full model. StructPrune reformulates the pruning objective with
a divide-and-conquer ADMM procedure, coordinates structured masks across
modules, and targets global pruning behavior with `O(sqrt(N))` GPU memory.

This repository provides pruning and evaluation entry points for OPT and LLaMA
families. It builds on the SparseGPT/SparseLLM style one-shot calibration
pipeline while adding structured pruning utilities, iterative correction, and
global alternating updates for MLP blocks.

## Method at a Glance

| Component | Role in the codebase |
| --- | --- |
| Calibration data | `datautils.py` samples WikiText2, PTB, or C4 segments |
| Model entry points | `opt_main.py` and `llama_main.py` load, prune, evaluate, and optionally save models |
| Global pruning loop | `model_utils.py` coordinates layer-wise capture, structured updates, and evaluation |
| Structured mask utilities | `pruning_utils.py` implements structured row/column masks, iterative correction, and SparseGPT-style pruning |
| Quantization helpers | `quant.py` provides optional low-bit quantization utilities |

## Installation

Create an environment and install the core dependencies:

```bash
git clone git@github.com:Hik289/structured_sparseGPT.git
cd structured_sparseGPT

python3 -m venv .venv
source .venv/bin/activate

pip install torch transformers datasets numpy pandas huggingface_hub wandb
```

The original experiments use GPU execution. Install the PyTorch build that
matches your CUDA version from the official PyTorch instructions if the generic
`pip install torch` command does not match your machine.

## Quickstart

Prune and evaluate a small OPT model:

```bash
python opt_main.py \
  --model facebook/opt-125m \
  --dataset c4 \
  --nsamples 128 \
  --sparsity 0.7 \
  --cudan cuda:0
```

Run LLaMA-style pruning:

```bash
python llama_main.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset c4 \
  --nsamples 32 \
  --sparsity 0.5 \
  --cudan cuda:0
```

Run semi-structured `N:M` pruning:

```bash
python opt_main.py \
  --model facebook/opt-125m \
  --dataset c4 \
  --prunen 2 \
  --prunem 4 \
  --cudan cuda:0
```

## Usage

### Main Arguments

| Argument | Description |
| --- | --- |
| `--model` | Hugging Face model identifier or local model path |
| `--dataset` | Calibration dataset: `wikitext2`, `ptb`, or `c4` |
| `--nsamples` | Number of calibration samples |
| `--sparsity` | Target sparsity ratio for pruning |
| `--prunen`, `--prunem` | Semi-structured `N:M` pruning pattern |
| `--percdamp` | Hessian dampening coefficient |
| `--blocksize` | Block size for adaptive mask selection |
| `--minlayer`, `--maxlayer` | Layer range to prune |
| `--prune_only` | Restrict pruning to layer names containing this string |
| `--invert` | Invert the layer-selection rule |
| `--gmp` | Run the magnitude-pruning baseline |
| `--wbits` | Optional quantization bit width |
| `--save` | Save the pruned model to a local path |
| `--cudan` | CUDA device string, e.g. `cuda:0` |

### Saving a Pruned Model

```bash
python opt_main.py \
  --model facebook/opt-125m \
  --dataset c4 \
  --sparsity 0.7 \
  --save checkpoints/opt125m_structprune \
  --cudan cuda:0
```

### Layer-Restricted Pruning

```bash
python llama_main.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset c4 \
  --sparsity 0.5 \
  --minlayer 4 \
  --maxlayer 24 \
  --prune_only mlp \
  --cudan cuda:0
```

## Repository Layout

```text
opt_main.py        OPT pruning and evaluation entry point
llama_main.py      LLaMA pruning and evaluation entry point
model_utils.py     Model loading, activation capture, global pruning loops
pruning_utils.py   Structured pruning, SparseGPT-style pruning, correction utilities
datautils.py       Calibration/evaluation data loading
quant.py           Quantization helpers
run.ipynb          Exploratory notebook
run2.ipynb         Exploratory notebook
LICENSE            Apache-2.0 license
```

## Notes

- The code downloads Hugging Face models and datasets as needed.
- Large models require substantial GPU memory; reduce `--nsamples` when memory
  is tight.
- LLaMA checkpoints may require Hugging Face access approval and local login.
- This implementation inherits ideas and utilities from
  [SparseGPT](https://arxiv.org/abs/2301.00774),
  [Wanda](https://arxiv.org/abs/2306.11695), and
  [SparseLLM](https://arxiv.org/abs/2402.17946).

## Citation

If you use this repository, please cite:

```bibtex
@misc{song2025structprunestructuredglobalpruning,
  title         = {StructPrune: Structured Global Pruning asymptotics with $\mathcal{O}(\sqrt{N})$ GPU Memory},
  author        = {Xinyuan Song and Guangji Bai and Liang Zhao},
  year          = {2025},
  eprint        = {2510.03246},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.03246}
}
```

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
