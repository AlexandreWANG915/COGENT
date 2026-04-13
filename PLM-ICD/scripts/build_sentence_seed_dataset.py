#!/usr/bin/env python3
"""
Prepare sentence-level seed data for GPT verification.

Sample 500 GT + 500 HN per code, prioritizing high prob samples.
"""
import json
import os
import random
from collections import defaultdict

def main():
    input_file = os.environ.get('EVIDENCE_FILE', 'data_v4/evidence_sentence_merged.jsonl')
    output_file = os.environ.get('OUTPUT_FILE', 'data_v4/seed_data/sentence_seed_samples.jsonl')

    # Load code descriptions
    code_desc_file = os.environ.get('CODE_DESC_FILE', 'data/icd10_code_descriptions.jsonl')
    code_descriptions = {}
    with open(code_desc_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            code_descriptions[data['code']] = data['description']
    print(f"Loaded {len(code_descriptions)} code descriptions")

    # Sampling parameters
    MAX_GT_PER_CODE = 500
    MAX_HN_PER_CODE = 500

    # Load evidence data
    print(f"Loading evidence from {input_file}...")
    records = []
    with open(input_file, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} records")

    # Group by code and is_gt
    code_groups = defaultdict(lambda: {'gt': [], 'hn': []})
    for record in records:
        code = record['code']
        if record.get('is_gt', False):
            code_groups[code]['gt'].append(record)
        else:
            code_groups[code]['hn'].append(record)

    print(f"  {len(code_groups)} unique codes")

    # Sample and prepare output
    print("\nSampling...")
    sampled_records = []
    stats = defaultdict(lambda: {'gt': 0, 'hn': 0})

    for code in sorted(code_groups.keys()):
        gt_records = code_groups[code]['gt']
        hn_records = code_groups[code]['hn']

        # Sort by prob descending (higher prob first)
        gt_records.sort(key=lambda x: x['pred_prob'], reverse=True)
        hn_records.sort(key=lambda x: x['pred_prob'], reverse=True)

        # Sample GT
        gt_sampled = gt_records[:MAX_GT_PER_CODE]
        stats[code]['gt'] = len(gt_sampled)

        # Sample HN
        hn_sampled = hn_records[:MAX_HN_PER_CODE]
        stats[code]['hn'] = len(hn_sampled)

        # Add to output with enriched info
        for record in gt_sampled + hn_sampled:
            # Get best sentence (highest score)
            sentences = record.get('sentences', [])
            if not sentences:
                continue

            best_sentence = max(sentences, key=lambda x: x.get('score', 0))

            sampled_records.append({
                'note_id': record['note_id'],
                'code': record['code'],
                'code_description': code_descriptions.get(record['code'], ''),
                'is_gt': record.get('is_gt', False),
                'prob': record['pred_prob'],
                'sentence_text': best_sentence['text'],
                'sentence_section': best_sentence.get('section', ''),
                'sentence_score': best_sentence.get('score', 0),
                'all_sentences': sentences  # Keep all for reference
            })

    # Print statistics
    print("\nPer-code statistics:")
    total_gt = 0
    total_hn = 0
    for code in sorted(stats.keys()):
        s = stats[code]
        print(f"  {code}: GT={s['gt']}, HN={s['hn']}")
        total_gt += s['gt']
        total_hn += s['hn']

    print(f"\nTotal: GT={total_gt}, HN={total_hn}, Total={total_gt + total_hn}")

    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        for record in sampled_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Done! Saved {len(sampled_records)} samples")


if __name__ == '__main__':
    main()
