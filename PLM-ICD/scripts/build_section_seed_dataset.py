#!/usr/bin/env python3
"""
Prepare section-level seed data for GPT verification.

Strategy:
- Use max attention section only (matches Qwen training format)
- Sample 500 GT + 500 HN per code (high prob first for reliability)
- Include section content for GPT verification
"""
import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import random

# Config
EVIDENCE_FILE = os.environ.get('EVIDENCE_FILE', 'data_v4/evidence_section.jsonl')
SECTION_FILE = os.environ.get('SECTION_FILE', 'data/mimiciv_icd10_with_sections_v4.feather')
CODE_DESC_FILE = os.environ.get('CODE_DESC_FILE', 'data/icd10_code_descriptions.jsonl')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'data_v4/seed_data')

SAMPLES_PER_CODE_GT = 500
SAMPLES_PER_CODE_HN = 500
MIN_SECTION_ATTENTION = 0.01  # Minimum attention score to consider


def load_code_descriptions(filepath):
    """Load ICD-10 code descriptions."""
    code_desc = {}
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            code_desc[record['code']] = record['description']
    return code_desc


def normalize_section_name(name):
    """Normalize section name for comparison."""
    return name.lower().replace(' ', '_')


def load_section_content(filepath):
    """Load section content from feather file."""
    print(f"Loading section content from {filepath}...")
    df = pd.read_feather(filepath)

    # Get all section columns
    section_cols = [c for c in df.columns if c.startswith('section_')]
    print(f"  Found {len(section_cols)} section columns")

    # Map column names -> normalized keys
    # section_history_of_present_illness -> history_of_present_illness
    col_to_normalized = {col: col.replace('section_', '') for col in section_cols}

    # Build note_id -> sections mapping (using normalized keys)
    note_sections = {}
    for _, row in df.iterrows():
        note_id = row['note_id']
        sections = {}

        # Collect all section content with normalized keys
        for col in section_cols:
            content = row.get(col)
            if content and isinstance(content, str) and len(content.strip()) > 0:
                normalized_key = col_to_normalized[col]
                sections[normalized_key] = content

        note_sections[note_id] = sections

    print(f"  Loaded sections for {len(note_sections)} notes")
    return note_sections


def get_max_attention_section(section_attention):
    """Get the section with highest attention score."""
    if not section_attention:
        return None, 0.0

    max_section = max(section_attention.items(), key=lambda x: x[1])
    return max_section[0], max_section[1]


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load code descriptions
    print("Loading code descriptions...")
    code_desc = load_code_descriptions(CODE_DESC_FILE)
    print(f"  Loaded {len(code_desc)} code descriptions")

    # Load section content
    note_sections = load_section_content(SECTION_FILE)

    # First pass: collect all candidates grouped by code
    print(f"\nLoading evidence from {EVIDENCE_FILE}...")
    code_candidates = defaultdict(lambda: {'gt': [], 'hn': []})

    with open(EVIDENCE_FILE, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0:
                print(f"  Processing line {line_num}...", end='\r')

            record = json.loads(line)
            note_id = record['note_id']

            for cand in record['candidates']:
                code = cand['code']
                prob = cand['prob']
                is_gt = cand.get('is_gt', False)
                section_attention = cand.get('section_attention', {})

                # Get max attention section
                max_section, max_score = get_max_attention_section(section_attention)

                if max_section is None or max_score < MIN_SECTION_ATTENTION:
                    continue

                candidate_info = {
                    'note_id': note_id,
                    'code': code,
                    'prob': prob,
                    'is_gt': is_gt,
                    'max_section': max_section,
                    'max_section_score': max_score,
                    'section_attention': section_attention
                }

                if is_gt:
                    code_candidates[code]['gt'].append(candidate_info)
                else:
                    code_candidates[code]['hn'].append(candidate_info)

    print(f"\nLoaded candidates for {len(code_candidates)} codes")

    # Statistics before sampling
    print("\nPre-sampling statistics:")
    total_gt = sum(len(v['gt']) for v in code_candidates.values())
    total_hn = sum(len(v['hn']) for v in code_candidates.values())
    print(f"  Total GT candidates: {total_gt:,}")
    print(f"  Total HN candidates: {total_hn:,}")

    # Sample candidates
    print(f"\nSampling {SAMPLES_PER_CODE_GT} GT + {SAMPLES_PER_CODE_HN} HN per code...")
    sampled_candidates = []

    for code in sorted(code_candidates.keys()):
        gt_list = code_candidates[code]['gt']
        hn_list = code_candidates[code]['hn']

        # Sort by prob descending (higher prob = more reliable attention)
        gt_list.sort(key=lambda x: x['prob'], reverse=True)
        hn_list.sort(key=lambda x: x['prob'], reverse=True)

        # Sample
        sampled_gt = gt_list[:SAMPLES_PER_CODE_GT]
        sampled_hn = hn_list[:SAMPLES_PER_CODE_HN]

        sampled_candidates.extend(sampled_gt)
        sampled_candidates.extend(sampled_hn)

        print(f"  {code}: GT={len(sampled_gt)}/{len(gt_list)}, HN={len(sampled_hn)}/{len(hn_list)}")

    print(f"\nTotal sampled: {len(sampled_candidates)} candidates")

    # Enrich with section content
    print("\nEnriching with section content...")
    enriched_samples = []
    missing_sections = 0
    missing_notes = 0

    for cand in sampled_candidates:
        note_id = cand['note_id']
        max_section = cand['max_section']
        normalized_section = normalize_section_name(max_section)

        if note_id not in note_sections:
            missing_notes += 1
            continue

        sections = note_sections[note_id]
        if normalized_section not in sections:
            missing_sections += 1
            continue

        section_content = sections[normalized_section]
        if not section_content or len(section_content.strip()) < 10:
            continue

        # Get code description
        desc = code_desc.get(cand['code'], 'Unknown')

        enriched_sample = {
            'note_id': note_id,
            'code': cand['code'],
            'code_description': desc,
            'prob': cand['prob'],
            'is_gt': cand['is_gt'],
            'section_name': max_section,
            'section_score': cand['max_section_score'],
            'section_content': section_content,
            'section_attention': cand['section_attention']
        }
        enriched_samples.append(enriched_sample)

    print(f"  Missing notes: {missing_notes}")
    print(f"  Missing sections: {missing_sections}")
    print(f"  Enriched samples: {len(enriched_samples)}")

    # Final statistics
    print("\nFinal statistics:")
    code_stats = defaultdict(lambda: {'gt': 0, 'hn': 0})
    for sample in enriched_samples:
        code = sample['code']
        if sample['is_gt']:
            code_stats[code]['gt'] += 1
        else:
            code_stats[code]['hn'] += 1

    gt_total = sum(v['gt'] for v in code_stats.values())
    hn_total = sum(v['hn'] for v in code_stats.values())
    print(f"  Total GT: {gt_total}")
    print(f"  Total HN: {hn_total}")
    print(f"  Codes covered: {len(code_stats)}")

    # Check low coverage codes
    print("\nLow coverage codes:")
    for code in ['Z23.', 'Y92.230', 'Y92.239', 'Y92.9']:
        if code in code_stats:
            stats = code_stats[code]
            print(f"  {code}: GT={stats['gt']}, HN={stats['hn']}")
        else:
            print(f"  {code}: NOT FOUND")

    # Shuffle and save
    random.seed(42)
    random.shuffle(enriched_samples)

    output_file = f"{OUTPUT_DIR}/section_seed_samples.jsonl"
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        for sample in enriched_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Done! Saved {len(enriched_samples)} samples")

    # Also save statistics
    stats_file = f"{OUTPUT_DIR}/section_seed_stats.json"
    stats = {
        'total_samples': len(enriched_samples),
        'total_gt': gt_total,
        'total_hn': hn_total,
        'codes_covered': len(code_stats),
        'samples_per_code_gt': SAMPLES_PER_CODE_GT,
        'samples_per_code_hn': SAMPLES_PER_CODE_HN,
        'code_stats': {k: dict(v) for k, v in code_stats.items()}
    }
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_file}")


if __name__ == '__main__':
    main()
