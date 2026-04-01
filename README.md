# COGENT: Bridging PLM Prediction and LLM Reasoning for Explainable ICD Coding

This is the official repository for the paper **"COGENT: Bridging PLM Prediction and LLM Reasoning for Explainable ICD Coding"** (under review at COLM 2026).

## Overview

Automated ICD coding is a fundamental task in healthcare informatics. Discriminative Pre-trained Language Model (PLM) based methods achieve high recall but operate as black boxes, while Large Language Models (LLMs) offer reasoning capabilities but are prone to hallucination and underperform in structured multi-label prediction.

**COGENT** is a collaborative framework that bridges PLM prediction and LLM reasoning through a coarse-to-fine pipeline. It operates in three stages:

<p align="center">
  <img src="assets/framework.png" width="90%" alt="COGENT Framework"/>
</p>

### Stage 1: Evidence Mining
- Extracts multi-granularity code-evidence mappings from PLM label-wise attention
- Operates at both **section-level** (broader clinical context) and **sentence-level** (specific clinical findings)
- Constructs positive examples (supported codes) and hard negatives (attention-guided false positives)

### Stage 2: Reasoning Construction
- Employs GPT-4o as a teacher to filter noisy mappings and generate reasoning traces on a balanced subset
- Applies a **consistency filter** that cross-validates attention-based evidence against the teacher LLM's clinical judgment
- Distills the teacher's reasoning capability into Qwen3-30B to scale to the full dataset
- Produces **RICE** (Reasoning for ICD Coding with Evidence), a large-scale fine-grained reasoning dataset

### Stage 3: Explainable Verification
- Fine-tunes Qwen3-4B-Instruct as a verifier on RICE using LoRA
- Uses **confidence-based routing**: high-confidence PLM predictions are accepted directly; uncertain predictions are routed to the verifier
- The verifier examines each uncertain code with its localized evidence and produces an accept-or-reject decision with a natural language reasoning trace

## RICE Dataset

RICE contains **563K code-evidence pairs** covering **5,445 unique ICD codes** across **47,716 clinical notes** from MIMIC-III. It exceeds existing evidence-annotated datasets (MDACE, CodiEsp-X, MedCodER) by 1-2 orders of magnitude.

| Property | Value |
|---|---|
| Total code-evidence pairs | 563K |
| Unique ICD codes | 5,445 |
| Clinical notes | 47,716 |
| Source | MIMIC-III discharge summaries |
| Evidence granularity | Section-level + Sentence-level |
| Includes reasoning traces | Yes |

## Main Results

Overall performance on MIMIC-III and MIMIC-IV (full codes):

| Method | Reasoning | MIMIC-III Mi-F1 | MIMIC-III Mi-P | MIMIC-III Mi-R | MIMIC-IV Mi-F1 | MIMIC-IV Mi-P | MIMIC-IV Mi-R |
|---|---|---|---|---|---|---|---|
| LAAT | No | 46.1 | 50.3 | 42.6 | 43.0 | 43.8 | 42.3 |
| CAML | No | 50.6 | 63.6 | 42.0 | 49.8 | 63.0 | 41.2 |
| PLM-ICD | No | 58.7 | 55.2 | 62.7 | 57.9 | 59.2 | 56.7 |
| Direct Prompting | No | 2.7 | 1.9 | 5.2 | 1.3 | 0.9 | 2.8 |
| Naive SFT | No | 30.4 | 24.4 | 40.4 | 46.4 | 45.5 | 47.4 |
| Tree Search | No | 10.6 | 10.9 | 10.3 | 11.6 | 10.2 | 13.3 |
| **COGENT (Ours)** | **Yes** | **59.5** | **58.0** | **61.2** | **58.4** | **60.7** | **56.3** |

COGENT achieves the highest Micro F1 on both datasets while being the only method that provides interpretable reasoning traces for each coding decision.

## Project Structure

```
COGENT/
├── README.md
├── requirements.txt
├── data/                        # Data preprocessing and splits
│   ├── mimic3/
│   └── mimic4/
├── plm/                         # PLM-ICD classifier (Stage 0)
│   ├── train.py
│   └── predict.py
├── evidence_mining/             # Stage 1: Evidence Mining
│   ├── extract_attention.py
│   └── build_candidates.py
├── reasoning_construction/      # Stage 2: Reasoning Construction
│   ├── teacher_refinement.py
│   ├── consistency_filter.py
│   └── distillation.py
├── verification/                # Stage 3: Explainable Verification
│   ├── train_verifier.py
│   └── inference.py
├── evaluation/                  # Evaluation scripts
│   └── metrics.py
└── configs/                     # Configuration files
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- Transformers >= 4.40
- DeepSpeed
- vLLM (for efficient LLM inference)
- Access to MIMIC-III / MIMIC-IV datasets (requires PhysioNet credentialed access)

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

Prepare MIMIC-III/IV discharge summaries following the preprocessing pipeline of [Mullenbach et al. (2018)](https://arxiv.org/abs/1802.05695):

```bash
python data/preprocess.py --dataset mimic3 --data_dir /path/to/mimic3
```

### 2. Train PLM-ICD Classifier

Train the PLM-ICD model with Longformer encoder:

```bash
python plm/train.py --dataset mimic3 --epochs 20 --optimizer adamw
```

### 3. Evidence Mining

Extract multi-granularity code-evidence mappings from PLM attention:

```bash
python evidence_mining/extract_attention.py \
    --model_path checkpoints/plm_icd \
    --dataset mimic3 \
    --top_k 100 \
    --top_m 5
```

### 4. Reasoning Construction

Run teacher refinement with GPT-4o and distillation:

```bash
# Teacher refinement (balanced subset)
python reasoning_construction/teacher_refinement.py \
    --candidates data/candidates.json \
    --output data/refined.json

# Distill to Qwen3-30B
python reasoning_construction/distillation.py \
    --refined_data data/refined.json \
    --base_model Qwen/Qwen3-30B \
    --output_dir checkpoints/distilled

# Scale to full dataset
python reasoning_construction/distillation.py \
    --mode inference \
    --model_path checkpoints/distilled \
    --candidates data/remaining_candidates.json \
    --output data/rice.json
```

### 5. Train Verifier

Fine-tune Qwen3-4B-Instruct with LoRA on RICE:

```bash
python verification/train_verifier.py \
    --base_model Qwen/Qwen3-4B-Instruct \
    --data_path data/rice.json \
    --lora_rank 8 \
    --epochs 5 \
    --lr 1e-4 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 8192 \
    --deepspeed configs/ds_zero3.json
```

### 6. Inference

Run the full COGENT pipeline:

```bash
python verification/inference.py \
    --plm_model checkpoints/plm_icd \
    --verifier_model checkpoints/verifier \
    --dataset mimic3 \
    --split test \
    --tau_ver 0.6
```

## Key Hyperparameters

| Stage | Parameter | Value |
|---|---|---|
| Evidence Mining | Top-K attention tokens | 100 |
| Evidence Mining | Top-M sentences | 5 |
| Evidence Mining | Hard negative threshold (tau_neg) | 0.3 |
| Evidence Mining | PLM decision threshold (tau_plm) | 0.3-0.4* |
| Reasoning Construction | Balanced samples per code (<100 candidates) | 10 |
| Reasoning Construction | Balanced samples per code (100-500 candidates) | 20 |
| Reasoning Construction | Balanced samples per code (>500 candidates) | 50 |
| Verification | Routing threshold (tau_ver) | 0.5-0.7* |

\* Tuned on the validation set to maximize micro-F1.

## Citation

```bibtex
@inproceedings{cogent2026,
  title={COGENT: Bridging PLM Prediction and LLM Reasoning for Explainable ICD Coding},
  author={Anonymous},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```

## License

This project is for research purposes only. Use of MIMIC data requires a signed data use agreement through [PhysioNet](https://physionet.org/).

## Acknowledgments

This work uses de-identified clinical data from MIMIC-III and MIMIC-IV, accessed under a credentialed PhysioNet data use agreement. GPT-4o is used via the Azure OpenAI Service for reasoning trace generation.
