# COGENT: Bridging PLM Prediction and LLM Reasoning for Explainable ICD Coding

This repository provides code for COGENT, a framework for explainable automatic ICD coding that combines a pretrained language model with LLM-based verification.

## Abstract

Automatic ICD coding requires both broad label coverage and reliable clinical grounding. COGENT follows a two-stage design: a PLM first proposes candidate codes and locates supporting evidence in the clinical note, and a verifier model then re-checks uncertain candidates using structured evidence. This design keeps the recall advantages of PLM-based coding while improving precision through explicit evidence-based verification.

## Overview

The repository is organized into two components:

- `PLM-ICD`: candidate generation, evidence mining, GPT-assisted seed verification, and SFT data construction.
- `LLaMA-Factory`: verifier inference built on top of LLaMA-Factory.

The overall pipeline is:

1. Train a PLM ICD coder.
2. Mine evidence from model attention.
3. Build and verify seed data with GPT.
4. Construct SFT datasets.
5. Run verifier inference on candidate codes.

## Repository Structure

### `PLM-ICD`

- `train_longformer_icd.sh`: Longformer training launcher.
- `src/train_longformer_icd.py`: PLM training entrypoint.
- `src/mine_section_evidence.py`: evidence mining from section-level note structure.
- `src/mine_sentence_evidence.py`: evidence mining from sentence-level note structure.
- `scripts/build_*_seed_dataset.py`: seed construction for GPT verification.
- `scripts/verify_*_seeds_with_gpt.py`: GPT-based seed verification.
- `scripts/build_*_sft.py`: SFT data generation for verifier or multitask training.

### `LLaMA-Factory`

- `run_verifier_inference.py`: verifier inference entrypoint.
- `run_verifier_inference.sh`: example launcher for verifier inference.

## Requirements

- Python >= 3.10
- PyTorch
- transformers
- datasets
- accelerate
- pandas
- numpy
- scikit-learn
- openai
- vllm

Install the verifier-side package with:

```bash
cd LLaMA-Factory
pip install -e .
```

## Data

This repository does not redistribute MIMIC data or derived clinical artifacts.

To run the pipeline, prepare:

- ICD training files for the PLM stage
- a structured-note feather file for evidence mining and verifier inference
- a JSONL file with ICD code descriptions

The PLM training stage expects note text and semicolon-separated ICD labels. The code description file should contain one JSON object per line:

```json
{"code": "I10", "description": "Essential (primary) hypertension"}
```

Intermediate artifacts are written under `PLM-ICD/data/` by default, including candidate files, seed datasets, verified seed files, and SFT datasets.

## Usage

### Train the PLM candidate generator

From `PLM-ICD`:

```bash
bash train_longformer_icd.sh 0 1 0 /path/to/output_dir
```

To override the default input paths:

```bash
MODEL_NAME_OR_PATH=/path/to/clinical-longformer \
TRAIN_FILE=/path/to/train.csv \
VALIDATION_FILE=/path/to/val.csv \
CODE_FILE=/path/to/ALL_CODES.txt \
bash train_longformer_icd.sh 0 1 0 /path/to/output_dir
```

Arguments:

1. GPU id
2. `use_laat` (`0` or `1`)
3. `from_scratch` (`0` or `1`)
4. output directory

### Mine evidence

Typical commands:

```bash
python src/mine_section_evidence.py ...
python src/mine_sentence_evidence.py ...
```

These scripts take a trained PLM checkpoint together with structured note data and produce evidence-linked candidate files.

### Build and verify seed datasets

Section-oriented flow:

```bash
python scripts/build_section_seed_dataset.py ...
python scripts/verify_section_seeds_with_gpt.py ...
```

Sentence-oriented flow:

```bash
python scripts/build_sentence_seed_dataset.py ...
python scripts/verify_sentence_seeds_with_gpt.py ...
```

For GPT verification, set:

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=...
```

Optional settings:

```bash
export AZURE_OPENAI_API_VERSION=2025-01-01-preview
export AZURE_OPENAI_MODEL=gpt-4o
```

### Build SFT datasets

Examples:

```bash
python scripts/build_section_verifier_sft.py ...
python scripts/build_sentence_verifier_sft.py ...
python scripts/build_section_multitask_sft.py ...
python scripts/build_sentence_multitask_sft.py ...
python scripts/build_evidence_tasks_sft.py ...
```

### Run verifier inference

From `LLaMA-Factory`:

```bash
cd LLaMA-Factory
FEATHER_FILE=/path/to/structured_notes.feather \
bash run_verifier_inference.sh
```

Common overrides:

```bash
PLM_ICD_ROOT=/path/to/PLM-ICD \
INPUT_FILE=/path/to/candidates.jsonl \
ADAPTER_PATH=/path/to/lora_adapter \
FEATHER_FILE=/path/to/structured_notes.feather \
bash run_verifier_inference.sh
```

## Notes

- Some scripts assume outputs from earlier stages already exist.
- `LLaMA-Factory/data/dataset_info.json` is kept close to upstream and was not rewritten here.
