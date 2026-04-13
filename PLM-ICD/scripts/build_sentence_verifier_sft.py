#!/usr/bin/env python3
"""
Prepare SFT data for Sentence-Level Verifier (Task 2 format).

Input: Sentence + Code
Output: <think>reasoning</think><answer>Yes/No</answer>

Uses GPT-verified sentence seed data with reasoning.
"""

import json
import random
import argparse
from collections import defaultdict


def load_seed_data(input_file: str) -> list:
    """Load GPT-verified sentence seed data."""
    records = []
    with open(input_file) as f:
        for line in f:
            record = json.loads(line)
            # Only include samples with final_label
            if record.get('final_label'):
                records.append(record)
    return records


def create_sft_sample(record: dict) -> dict:
    """Create a single SFT sample from a sentence record.

    Input format:
        You are a medical coding specialist. Determine if the clinical evidence
        supports the ICD-10 diagnosis code.

        Code: {code} - {description}

        Section: {section}
        Evidence: {sentence_text}

    Output format:
        <think>reasoning</think>
        <answer>Yes/No</answer>
    """
    code = record['code']
    description = record.get('code_description', '')
    section = record.get('sentence_section', '')
    sentence = record.get('sentence_text', '')
    think = record.get('gpt_think', '')

    # Determine answer based on final_label
    answer = 'Yes' if record.get('final_label') == 'positive' else 'No'

    # Build input prompt
    input_text = f"""You are a medical coding specialist. Determine if the clinical evidence supports the ICD-10 diagnosis code.

Code: {code} - {description}

Section: {section}
Evidence: {sentence}

Determine if this evidence supports the diagnosis code:
- Answer "Yes" if there is direct mention, direct evidence, or reasonable clinical inference
- Answer "No" if there is no evidence or mention of the diagnosis

Output format:
<think>Your reasoning about whether the evidence supports the code</think>
<answer>Yes or No</answer>"""

    # Build output
    output_text = f"<think>{think}</think>\n<answer>{answer}</answer>"

    return {
        "conversations": [
            {"from": "human", "value": input_text},
            {"from": "gpt", "value": output_text}
        ],
        "_meta": {
            "note_id": record['note_id'],
            "code": code,
            "is_gt": record.get('is_gt', False),
            "label": answer,
            "section": section
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for Sentence-Level Verifier")
    parser.add_argument("--input", type=str,
                        default="data_v4/seed_data/sentence_seed_final.jsonl",
                        help="Input GPT-verified sentence seed data")
    parser.add_argument("--output-train", type=str,
                        default="data_v4/seed_data/sft_sentence_verifier_train.json",
                        help="Training output file")
    parser.add_argument("--output-val", type=str,
                        default="data_v4/seed_data/sft_sentence_verifier_val.json",
                        help="Validation output file")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load seed data
    print(f"Loading seed data from {args.input}...")
    records = load_seed_data(args.input)
    print(f"  Loaded {len(records)} records")

    # Statistics
    label_counts = defaultdict(int)
    section_counts = defaultdict(int)
    code_counts = defaultdict(int)

    for r in records:
        label_counts[r.get('final_label', 'unknown')] += 1
        section_counts[r.get('sentence_section', 'unknown')] += 1
        code_counts[r['code']] += 1

    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print(f"\nSection distribution (top 6):")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1])[:6]:
        print(f"  {section}: {count}")

    print(f"\nCodes: {len(code_counts)}")

    # Create SFT samples
    print("\nCreating SFT samples...")
    samples = [create_sft_sample(r) for r in records]
    print(f"  Created {len(samples)} samples")

    # Shuffle and split
    random.shuffle(samples)
    val_size = int(len(samples) * args.val_ratio)
    train_samples = samples[val_size:]
    val_samples = samples[:val_size]

    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")

    # Statistics for train set
    train_yes = sum(1 for s in train_samples if s['_meta']['label'] == 'Yes')
    train_no = len(train_samples) - train_yes
    print(f"  Train - Yes: {train_yes}, No: {train_no}")

    val_yes = sum(1 for s in val_samples if s['_meta']['label'] == 'Yes')
    val_no = len(val_samples) - val_yes
    print(f"  Val - Yes: {val_yes}, No: {val_no}")

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
    print(f"  Train: {len(train_samples)} (Yes: {train_yes}, No: {train_no})")
    print(f"  Val: {len(val_samples)} (Yes: {val_yes}, No: {val_no})")


if __name__ == "__main__":
    main()
