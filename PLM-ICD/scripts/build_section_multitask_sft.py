#!/usr/bin/env python3
"""
Prepare Section-Level SFT Data for Task 1 (Code Prediction) and Task 2 (Verification).

Data sources:
- Model: section_verified_by_model.jsonl (is_gt=True), section_discarded_by_model.jsonl (is_gt=False)
- GPT: seed_data/section_seed_final.jsonl (gpt_status=success)

Task 2: Verification samples (Model + GPT combined)
Task 1: Code prediction samples (GT without HN)
"""

# Fixed validation set size
VAL_SIZE = 2000

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


def load_jsonl(filepath: str) -> list:
    """Load JSONL file."""
    records = []
    with open(filepath) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def group_by_section(records: list) -> dict:
    """Group records by (note_id, section_name)."""
    groups = defaultdict(list)
    for record in records:
        key = (record['note_id'], record['section_name'])
        groups[key].append(record)
    return groups


def normalize_gpt_record(record: dict) -> dict:
    """Normalize GPT record to match model record format.

    GPT fields mapping:
    - gpt_valid -> _verification.verified
    - gpt_think -> _verification.reasoning
    - gpt_answer -> _verification.answer
    """
    normalized = record.copy()
    normalized['_verification'] = {
        'verified': record.get('gpt_valid', False),
        'reasoning': record.get('gpt_think', ''),
        'answer': record.get('gpt_answer', '')
    }
    return normalized


def create_task2_sample(section_records: list, source: str = "model") -> dict:
    """Create Task 2 (Verification) sample from GT + HN records.

    Input: Section content + Candidate codes
    Output: <code><think><answer> triplets for each code

    Args:
        section_records: List of records for this section
        source: Data source ("model" or "gpt")
    """
    if not section_records:
        return None

    first = section_records[0]
    section_name = first['section_name']
    section_content = first['section_content']
    note_id = first['note_id']

    # Build candidate list and output
    candidates = []
    outputs = []
    code_labels = []

    for record in section_records:
        code = record['code']
        description = record.get('code_description', '')

        # Get reasoning from _verification
        verification = record.get('_verification', {})
        think = verification.get('reasoning', '')
        is_verified = verification.get('verified', False)
        answer = 'Yes' if is_verified else 'No'

        candidates.append(f"- {code} - {description}")
        outputs.append(f"<code>{code}</code>\n<think>{think}</think>\n<answer>{answer}</answer>")
        code_labels.append({
            "code": code,
            "label": answer,
            "is_gt": record.get('is_gt', False)
        })

    candidate_text = "\n".join(candidates)
    input_text = f"""You are a medical coding specialist. Given the following section of a discharge summary, determine which ICD-10 codes are supported by the text.

Section: {section_name}

Content:
{section_content}

Candidate codes to verify:
{candidate_text}

For each code, determine if it is supported by THIS section's content:
- Answer "Yes" if there is direct mention, direct evidence, or reasonable clinical inference
- Answer "No" if there is no evidence or mention of the diagnosis

For each code, output in the following format:
<code>CODE_HERE</code>
<think>Your reasoning about whether the evidence supports this code</think>
<answer>Yes or No</answer>"""

    output_text = "\n\n".join(outputs)

    return {
        "task": "verification",
        "source": source,
        "conversations": [
            {"from": "human", "value": input_text},
            {"from": "gpt", "value": output_text}
        ],
        "_meta": {
            "note_id": note_id,
            "section_name": section_name,
            "num_codes": len(section_records),
            "codes": [r['code'] for r in section_records],
            "code_labels": code_labels,
            "num_positive": sum(1 for cl in code_labels if cl['label'] == 'Yes'),
            "num_negative": sum(1 for cl in code_labels if cl['label'] == 'No')
        }
    }


def create_task1_sample(section_key: tuple, gt_records: list, source: str = "model") -> dict:
    """Create Task 1 (Code Prediction) sample from GT records only.

    Input: Section content (evidence)
    Output: Comma-separated ICD codes

    Args:
        section_key: Tuple of (note_id, section_name)
        gt_records: List of GT records for this section
        source: Data source ("model" or "gpt")
    """
    if not gt_records:
        return None

    note_id, section_name = section_key
    first = gt_records[0]
    section_content = first['section_content']

    # Collect all codes
    codes = [r['code'] for r in gt_records]

    input_text = f"""You are a medical coding specialist. Based on the following clinical evidence, output the ICD-10 diagnosis codes.

Evidence: {section_content}"""

    output_text = f"<code>{', '.join(codes)}</code>"

    return {
        "task": "code_prediction",
        "source": source,
        "conversations": [
            {"from": "human", "value": input_text},
            {"from": "gpt", "value": output_text}
        ],
        "_meta": {
            "note_id": note_id,
            "section_name": section_name,
            "num_codes": len(codes),
            "codes": codes
        }
    }


def save_json(data: list, filepath: str):
    """Save data to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare Section-Level SFT Data")
    parser.add_argument("--verified-file", type=str,
                        default="data_v4/section_verified_by_model.jsonl",
                        help="Verified data file (GT source)")
    parser.add_argument("--discarded-file", type=str,
                        default="data_v4/section_discarded_by_model.jsonl",
                        help="Discarded data file (HN source)")
    parser.add_argument("--seed-file", type=str,
                        default="data_v4/seed_data/section_seed_final.jsonl",
                        help="GPT seed data file")
    parser.add_argument("--output-dir", type=str,
                        default="data_v4/sft_section",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Step 1: Load data
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)

    print(f"Loading verified data from {args.verified_file}...")
    verified_records = load_jsonl(args.verified_file)
    print(f"  Total verified records: {len(verified_records)}")

    print(f"Loading discarded data from {args.discarded_file}...")
    discarded_records = load_jsonl(args.discarded_file)
    print(f"  Total discarded records: {len(discarded_records)}")

    # Load GPT seed data (optional)
    gpt_records = []
    if Path(args.seed_file).exists():
        print(f"Loading GPT seed data from {args.seed_file}...")
        gpt_records_raw = load_jsonl(args.seed_file)
        # Filter GPT records: only keep successful ones
        gpt_records = [r for r in gpt_records_raw if r.get('gpt_status') == 'success']
        # Normalize GPT records to match model format
        gpt_records = [normalize_gpt_record(r) for r in gpt_records]
        print(f"  Total GPT records: {len(gpt_records_raw)}, Success: {len(gpt_records)}")
    else:
        print(f"GPT seed file not found at {args.seed_file}, skipping GPT data")

    # Filter GT and HN from model data
    gt_records = [r for r in verified_records if r.get('is_gt') == True]
    hn_records = [r for r in discarded_records if r.get('is_gt') == False]

    print(f"\n  Model GT records (verified, is_gt=True): {len(gt_records)}")
    print(f"  Model HN records (discarded, is_gt=False): {len(hn_records)}")
    print(f"  GPT records (normalized): {len(gpt_records)}")

    # Step 2: Group by section
    print("\n" + "=" * 60)
    print("Step 2: Grouping by section")
    print("=" * 60)

    gt_by_section = group_by_section(gt_records)
    hn_by_section = group_by_section(hn_records)

    print(f"  Unique GT sections: {len(gt_by_section)}")
    print(f"  Unique HN sections: {len(hn_by_section)}")

    # Step 3: Build Task 2 (sections with HN) from Model data
    print("\n" + "=" * 60)
    print("Step 3: Building Task 2 (Verification) samples from Model data")
    print("=" * 60)

    task2_samples_model = []
    task2_sections = set()  # Track sections used for Task 2
    task2_gt_count = 0
    task2_hn_count = 0

    for section_key in hn_by_section.keys():
        task2_sections.add(section_key)

        gt_list = gt_by_section.get(section_key, [])
        hn_list = hn_by_section[section_key]

        # Combine GT + HN for this section
        combined = gt_list + hn_list
        sample = create_task2_sample(combined, source="model")

        if sample:
            task2_samples_model.append(sample)
            task2_gt_count += len(gt_list)
            task2_hn_count += len(hn_list)

    print(f"  Model Task 2 samples: {len(task2_samples_model)}")
    print(f"  Model Task 2 GT codes: {task2_gt_count}")
    print(f"  Model Task 2 HN codes: {task2_hn_count}")

    # Step 3b: Build Task 2 from GPT data (each record is a single sample)
    print("\n" + "=" * 60)
    print("Step 3b: Building Task 2 (Verification) samples from GPT data")
    print("=" * 60)

    gpt_by_section = group_by_section(gpt_records)
    task2_samples_gpt = []
    gpt_positive_count = 0
    gpt_negative_count = 0

    for section_key, records in gpt_by_section.items():
        sample = create_task2_sample(records, source="gpt")
        if sample:
            task2_samples_gpt.append(sample)
            gpt_positive_count += sample['_meta']['num_positive']
            gpt_negative_count += sample['_meta']['num_negative']

    print(f"  GPT Task 2 samples: {len(task2_samples_gpt)}")
    print(f"  GPT positive codes: {gpt_positive_count}")
    print(f"  GPT negative codes: {gpt_negative_count}")

    # Combine Model + GPT Task 2 samples
    task2_samples = task2_samples_model + task2_samples_gpt
    print(f"\n  Total Task 2 samples: {len(task2_samples)}")

    # Step 4: Build Task 1 (remaining sections without HN) from Model data only
    print("\n" + "=" * 60)
    print("Step 4: Building Task 1 (Code Prediction) samples")
    print("=" * 60)

    task1_samples = []
    task1_gt_count = 0

    for section_key, gt_list in gt_by_section.items():
        if section_key not in task2_sections:
            sample = create_task1_sample(section_key, gt_list, source="model")
            if sample:
                task1_samples.append(sample)
                task1_gt_count += len(gt_list)

    print(f"  Task 1 samples: {len(task1_samples)}")
    print(f"  Task 1 GT codes: {task1_gt_count}")

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    total_gt_used = task1_gt_count + task2_gt_count
    print(f"  Total GT from files: {len(gt_records)}")
    print(f"  GT used (Task1 + Task2): {total_gt_used}")
    print(f"  Match: {total_gt_used == len(gt_records)}")

    # Step 5: Train/Val split and save
    print("\n" + "=" * 60)
    print("Step 5: Splitting and saving (VAL_SIZE={})".format(VAL_SIZE))
    print("=" * 60)

    output_dir = Path(args.output_dir)

    # Task 1: Use fixed VAL_SIZE
    random.shuffle(task1_samples)
    val_size_t1 = min(VAL_SIZE, len(task1_samples))
    task1_train = task1_samples[val_size_t1:]
    task1_val = task1_samples[:val_size_t1]

    save_json(task1_train, output_dir / "task1_code_pred_train.json")
    save_json(task1_val, output_dir / "task1_code_pred_val.json")
    print(f"  Task 1 Train: {len(task1_train)}, Val: {len(task1_val)}")

    # Task 2: Use fixed VAL_SIZE
    random.shuffle(task2_samples)
    val_size_t2 = min(VAL_SIZE, len(task2_samples))
    task2_train = task2_samples[val_size_t2:]
    task2_val = task2_samples[:val_size_t2]

    save_json(task2_train, output_dir / "task2_verify_train.json")
    save_json(task2_val, output_dir / "task2_verify_val.json")
    print(f"  Task 2 Train: {len(task2_train)}, Val: {len(task2_val)}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Task 1 (Code Prediction):")
    print(f"  - Samples: {len(task1_samples)} ({len(task1_train)} train, {len(task1_val)} val)")
    print(f"  - Source: model only")
    print(f"  - GT codes: {task1_gt_count}")
    print(f"\nTask 2 (Verification):")
    print(f"  - Samples: {len(task2_samples)} ({len(task2_train)} train, {len(task2_val)} val)")
    print(f"  - Source breakdown:")
    print(f"    - Model: {len(task2_samples_model)} samples (GT: {task2_gt_count}, HN: {task2_hn_count})")
    print(f"    - GPT: {len(task2_samples_gpt)} samples (pos: {gpt_positive_count}, neg: {gpt_negative_count})")
    print(f"\nOutput directory: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
