#!/usr/bin/env python3
"""
Prepare SFT data for Section-Level Multi-Code Verifier (Task 3).

Input: Section content + Candidate codes
Output: Multiple <code><think><answer> triplets

Uses GPT-verified section seed data with reasoning.
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


def load_verified_data(input_file: str) -> list:
    """Load GPT-verified section data."""
    records = []
    with open(input_file) as f:
        for line in f:
            record = json.loads(line)
            # Only include successfully verified samples
            if record.get('gpt_status') == 'success' and record.get('gpt_valid') is not None:
                records.append(record)
    return records


def group_by_section(records: list) -> dict:
    """Group records by (note_id, section_name)."""
    groups = defaultdict(list)
    for record in records:
        key = (record['note_id'], record['section_name'])
        groups[key].append(record)
    return groups


def create_sft_sample(section_records: list) -> dict:
    """Create a single SFT sample from records sharing the same section.

    Input format:
        Section: {section_name}
        Content: {section_content}

        Candidate codes to verify:
        - CODE1 - Description1
        - CODE2 - Description2
        ...

    Output format:
        <code>CODE1</code>
        <think>reasoning</think>
        <answer>Yes/No</answer>

        <code>CODE2</code>
        <think>reasoning</think>
        <answer>Yes/No</answer>
        ...
    """
    if not section_records:
        return None

    # Get section info from first record
    first = section_records[0]
    section_name = first['section_name']
    section_content = first['section_content']
    note_id = first['note_id']

    # Build candidate list and output
    candidates = []
    outputs = []
    code_labels = []  # Track label for each code

    for record in section_records:
        code = record['code']
        description = record.get('code_description', '')
        think = record.get('gpt_think', '')
        answer = 'Yes' if record.get('gpt_valid') else 'No'

        candidates.append(f"- {code} - {description}")
        outputs.append(f"<code>{code}</code>\n<think>{think}</think>\n<answer>{answer}</answer>")
        code_labels.append({"code": code, "label": answer, "is_gt": record.get('is_gt', False)})

    # Build input prompt
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

    # Build output
    output_text = "\n\n".join(outputs)

    return {
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


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for Section-Level Verifier")
    parser.add_argument("--input", type=str,
                        default="data_v4/seed_data/section_seed_final.jsonl",
                        help="Input filtered section data (consistent samples only)")
    parser.add_argument("--output-train", type=str,
                        default="data_v4/seed_data/sft_section_verifier_train.json",
                        help="Training output file")
    parser.add_argument("--output-val", type=str,
                        default="data_v4/seed_data/sft_section_verifier_val.json",
                        help="Validation output file")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load verified data
    print(f"Loading verified data from {args.input}...")
    records = load_verified_data(args.input)
    print(f"  Loaded {len(records)} verified records")

    # Group by section
    print("Grouping by (note_id, section_name)...")
    groups = group_by_section(records)
    print(f"  {len(groups)} unique sections")

    # Create SFT samples
    print("Creating SFT samples...")
    samples = []
    for key, section_records in groups.items():
        sample = create_sft_sample(section_records)
        if sample:
            samples.append(sample)

    print(f"  Created {len(samples)} samples")

    # Statistics
    num_codes_dist = defaultdict(int)
    for s in samples:
        num_codes_dist[s['_meta']['num_codes']] += 1

    print("\nCodes per sample distribution:")
    for n in sorted(num_codes_dist.keys()):
        print(f"  {n} codes: {num_codes_dist[n]} samples")

    # Shuffle and split
    random.shuffle(samples)
    val_size = int(len(samples) * args.val_ratio)
    train_samples = samples[val_size:]
    val_samples = samples[:val_size]

    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")

    # Remove _meta for final output (optional, keep for debugging)
    # train_samples = [{k: v for k, v in s.items() if k != '_meta'} for s in train_samples]
    # val_samples = [{k: v for k, v in s.items() if k != '_meta'} for s in val_samples]

    # Save
    print(f"\nSaving to {args.output_train}...")
    with open(args.output_train, 'w') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)

    print(f"Saving to {args.output_val}...")
    with open(args.output_val, 'w') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)

    print("\nDone!")
    print(f"\nSummary:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")


if __name__ == "__main__":
    main()
