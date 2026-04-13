#!/usr/bin/env python3
"""
Prepare SFT data for Sentence-Level LLM training.

Task 1: Evidence → Code (Code Prediction)
Task 2: Evidence + Code → Yes/No + Reasoning (Verification)

Strategy:
- Task 2: Keep all HN, match GT to HN per code (balanced)
- Task 1: Remaining GT samples
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


def load_code_descriptions(path: str) -> dict:
    """Load ICD-10 code descriptions from JSONL file."""
    code_desc = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if 'code' in item and 'description' in item:
                code_desc[item['code']] = item['description']
    return code_desc


def format_evidence(sentences: list) -> str:
    """Format sentences into evidence text."""
    texts = []
    for sent in sentences:
        text = sent.get('text', '').strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def create_task1_sample(record: dict, code_desc: dict) -> dict:
    """Create Task 1 sample: Evidence → Code."""
    code = record['code']
    evidence = format_evidence(record.get('sentences', []))

    return {
        "task": "code_prediction",
        "source": "verifier",
        "conversations": [
            {
                "from": "human",
                "value": f"You are a medical coding specialist. Based on the following clinical evidence, predict the most appropriate ICD-10 diagnosis code.\n\nEvidence: {evidence}\n\nIMPORTANT: Output ONLY the code."
            },
            {
                "from": "gpt",
                "value": f"<code>{code}</code>"
            }
        ]
    }


def create_task2_sample(record: dict, code_desc: dict, label: str) -> dict:
    """Create Task 2 sample: Evidence + Code → Yes/No + Reasoning."""
    code = record['code']
    description = code_desc.get(code, "Unknown")
    evidence = format_evidence(record.get('sentences', []))

    # Get reasoning from verification result
    verification = record.get('_verification', {})
    reasoning = verification.get('reasoning', '')

    if not reasoning:
        # Fallback if no reasoning
        if label == "Yes":
            reasoning = "The clinical evidence supports this diagnosis code."
        else:
            reasoning = "The clinical evidence does not support this diagnosis code."

    return {
        "task": "verification",
        "source": "verifier",
        "conversations": [
            {
                "from": "human",
                "value": f"You are a medical coding specialist. Determine if the clinical evidence supports the ICD-10 diagnosis code.\n\nCode: {code} - {description}\n\nEvidence: {evidence}"
            },
            {
                "from": "gpt",
                "value": f"<think>{reasoning}</think>\n<answer>{label}</answer>"
            }
        ]
    }


def create_task1_sample_from_gpt(record: dict, code_desc: dict) -> dict:
    """Create Task 1 sample from GPT verified data."""
    code = record['code']
    evidence = record.get('sentence_text', '').strip()

    return {
        "task": "code_prediction",
        "source": "gpt",
        "conversations": [
            {
                "from": "human",
                "value": f"You are a medical coding specialist. Based on the following clinical evidence, predict the most appropriate ICD-10 diagnosis code.\n\nEvidence: {evidence}\n\nIMPORTANT: Output ONLY the code."
            },
            {
                "from": "gpt",
                "value": f"<code>{code}</code>"
            }
        ]
    }


def create_task2_sample_from_gpt(record: dict, code_desc: dict) -> dict:
    """Create Task 2 sample from GPT verified data."""
    code = record['code']
    description = code_desc.get(code, record.get('code_description', 'Unknown'))
    evidence = record.get('sentence_text', '').strip()
    reasoning = record.get('gpt_think', '')
    answer = record.get('gpt_answer', 'Yes')

    return {
        "task": "verification",
        "source": "gpt",
        "conversations": [
            {
                "from": "human",
                "value": f"You are a medical coding specialist. Determine if the clinical evidence supports the ICD-10 diagnosis code.\n\nCode: {code} - {description}\n\nEvidence: {evidence}"
            },
            {
                "from": "gpt",
                "value": f"<think>{reasoning}</think>\n<answer>{answer}</answer>"
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for Sentence-Level LLM")
    parser.add_argument("--input-verified", type=str,
                        default="data_v4/sentence_verified_by_model.jsonl",
                        help="Input verified evidence file (for Yes samples)")
    parser.add_argument("--input-discarded", type=str,
                        default="data_v4/sentence_discarded_by_model.jsonl",
                        help="Input discarded evidence file (for No samples)")
    parser.add_argument("--input-gpt", type=str,
                        default="data_v4/seed_data/sentence_seed_final.jsonl",
                        help="GPT verified seed data")
    parser.add_argument("--code-desc", type=str,
                        default="data/icd10_code_descriptions.jsonl",
                        help="Code description file")
    parser.add_argument("--output-dir", type=str,
                        default="data_v4/sft_sentence",
                        help="Output directory")
    parser.add_argument("--val-size", type=int, default=2000,
                        help="Validation set size per task")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load code descriptions
    print("Loading code descriptions...")
    code_desc = load_code_descriptions(args.code_desc)
    print(f"  Loaded {len(code_desc)} code descriptions")

    # Load verified GT (for Yes samples and Task 1)
    print(f"Loading verified data from {args.input_verified}...")
    gt_by_code = defaultdict(list)

    with open(args.input_verified) as f:
        for line in f:
            record = json.loads(line)
            code = record['code']
            if record.get('is_gt', True):
                gt_by_code[code].append(record)

    total_gt = sum(len(v) for v in gt_by_code.values())
    print(f"  Verified GT samples: {total_gt} ({len(gt_by_code)} codes)")

    # Load discarded HN (for No samples - model says "not supported")
    print(f"Loading discarded data from {args.input_discarded}...")
    hn_by_code = defaultdict(list)

    with open(args.input_discarded) as f:
        for line in f:
            record = json.loads(line)
            code = record['code']
            if not record.get('is_gt', True):  # Only HN
                hn_by_code[code].append(record)

    total_hn = sum(len(v) for v in hn_by_code.values())
    print(f"  Discarded HN samples: {total_hn} ({len(hn_by_code)} codes)")

    # Strategy: Keep all HN, match GT to HN per code
    print("\nBalancing Task 2 samples (GT matches HN per code)...")
    task2_gt = []
    task2_hn = []
    task1_gt = []

    codes_with_hn = 0
    codes_without_hn = 0

    for code in sorted(gt_by_code.keys()):
        gt_list = gt_by_code[code]
        hn_list = hn_by_code.get(code, [])
        hn_count = len(hn_list)

        if hn_count == 0:
            # No HN for this code, all GT goes to Task 1
            task1_gt.extend(gt_list)
            codes_without_hn += 1
        else:
            codes_with_hn += 1
            # Keep all HN
            task2_hn.extend(hn_list)

            # Sample GT to match HN count
            random.shuffle(gt_list)
            if hn_count >= len(gt_list):
                # More HN than GT, use all GT
                task2_gt.extend(gt_list)
            else:
                # Sample GT to match HN
                task2_gt.extend(gt_list[:hn_count])
                task1_gt.extend(gt_list[hn_count:])

    print(f"  Codes with HN: {codes_with_hn}")
    print(f"  Codes without HN: {codes_without_hn}")
    print(f"  Task 2 GT samples: {len(task2_gt)}")
    print(f"  Task 2 HN samples: {len(task2_hn)}")
    print(f"  Task 1 GT samples: {len(task1_gt)}")

    # Load GPT verified data
    gpt_task1_samples = []
    gpt_task2_samples = []
    if args.input_gpt and Path(args.input_gpt).exists():
        print(f"\nLoading GPT verified data from {args.input_gpt}...")
        gpt_pos = 0
        gpt_neg = 0
        with open(args.input_gpt) as f:
            for line in f:
                record = json.loads(line)
                answer = record.get('gpt_answer', 'Yes')

                # Task 2: All GPT samples go to verification
                gpt_task2_samples.append(create_task2_sample_from_gpt(record, code_desc))

                # Task 1: Only positive samples go to code prediction
                if answer == 'Yes':
                    gpt_task1_samples.append(create_task1_sample_from_gpt(record, code_desc))
                    gpt_pos += 1
                else:
                    gpt_neg += 1

        print(f"  GPT Task 1 samples: {len(gpt_task1_samples)}")
        print(f"  GPT Task 2 samples: {len(gpt_task2_samples)} (pos: {gpt_pos}, neg: {gpt_neg})")

    # Create Task 1 samples
    print("\nCreating Task 1 samples (Code Prediction)...")
    task1_samples = [create_task1_sample(r, code_desc) for r in task1_gt]
    # Merge with GPT samples
    task1_samples.extend(gpt_task1_samples)
    random.shuffle(task1_samples)
    print(f"  Total Task 1 samples: {len(task1_samples)} (verifier: {len(task1_gt)}, gpt: {len(gpt_task1_samples)})")

    # Split Task 1 train/val
    task1_val_size = min(args.val_size, len(task1_samples) // 2)
    task1_train = task1_samples[task1_val_size:]
    task1_val = task1_samples[:task1_val_size]
    print(f"  Task 1 train: {len(task1_train)}")
    print(f"  Task 1 val: {len(task1_val)}")

    # Create Task 2 samples
    print("\nCreating Task 2 samples (Verification)...")
    task2_samples = []
    for r in task2_gt:
        task2_samples.append(create_task2_sample(r, code_desc, "Yes"))
    for r in task2_hn:
        task2_samples.append(create_task2_sample(r, code_desc, "No"))
    verifier_task2_count = len(task2_samples)
    # Merge with GPT samples
    task2_samples.extend(gpt_task2_samples)
    random.shuffle(task2_samples)
    print(f"  Total Task 2 samples: {len(task2_samples)} (verifier: {verifier_task2_count}, gpt: {len(gpt_task2_samples)})")

    # Split Task 2 train/val
    task2_val_size = min(args.val_size, len(task2_samples) // 2)
    task2_train = task2_samples[task2_val_size:]
    task2_val = task2_samples[:task2_val_size]
    print(f"  Task 2 train: {len(task2_train)}")
    print(f"  Task 2 val: {len(task2_val)}")

    # Verify no overlap
    print("\nVerifying no overlap...")
    task1_keys = set()
    for r in task1_gt:
        task1_keys.add((r['note_id'], r['code']))

    task2_gt_keys = set()
    for r in task2_gt:
        task2_gt_keys.add((r['note_id'], r['code']))

    overlap = task1_keys & task2_gt_keys
    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping samples found!")
    else:
        print("  No overlap found. Good!")

    # Task 2 balance check
    print("\nTask 2 balance check:")
    yes_count = sum(1 for s in task2_samples if '"Yes"' in s['conversations'][1]['value'] or '<answer>Yes</answer>' in s['conversations'][1]['value'])
    no_count = len(task2_samples) - yes_count
    print(f"  Yes (GT): {yes_count}")
    print(f"  No (HN): {no_count}")
    print(f"  Ratio: {yes_count/max(no_count,1):.2f}")

    # Task 2 code distribution
    print("\nTask 2 code distribution (top 10):")
    code_counts = defaultdict(lambda: {'yes': 0, 'no': 0})
    for r in task2_gt:
        code_counts[r['code']]['yes'] += 1
    for r in task2_hn:
        code_counts[r['code']]['no'] += 1

    sorted_codes = sorted(code_counts.items(), key=lambda x: x[1]['yes'] + x[1]['no'], reverse=True)
    for code, counts in sorted_codes[:10]:
        print(f"  {code}: Yes={counts['yes']}, No={counts['no']}, Total={counts['yes']+counts['no']}")

    # Save outputs
    print("\nSaving outputs...")

    task1_train_path = output_dir / "task1_code_pred_train.json"
    with open(task1_train_path, 'w') as f:
        json.dump(task1_train, f, indent=2, ensure_ascii=False)
    print(f"  Saved {task1_train_path}")

    task1_val_path = output_dir / "task1_code_pred_val.json"
    with open(task1_val_path, 'w') as f:
        json.dump(task1_val, f, indent=2, ensure_ascii=False)
    print(f"  Saved {task1_val_path}")

    task2_train_path = output_dir / "task2_verify_train.json"
    with open(task2_train_path, 'w') as f:
        json.dump(task2_train, f, indent=2, ensure_ascii=False)
    print(f"  Saved {task2_train_path}")

    task2_val_path = output_dir / "task2_verify_val.json"
    with open(task2_val_path, 'w') as f:
        json.dump(task2_val, f, indent=2, ensure_ascii=False)
    print(f"  Saved {task2_val_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Task 1 (Code Prediction): {len(task1_train) + len(task1_val)} samples")
    print(f"  Train: {len(task1_train)}")
    print(f"  Val: {len(task1_val)}")
    print(f"  Sources: verifier={len(task1_gt)}, gpt={len(gpt_task1_samples)}")
    print(f"\nTask 2 (Verification): {len(task2_train) + len(task2_val)} samples")
    print(f"  Train: {len(task2_train)}")
    print(f"  Val: {len(task2_val)}")
    print(f"  Verifier: Yes={len(task2_gt)}, No={len(task2_hn)}")
    print(f"  GPT: {len(gpt_task2_samples)}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
