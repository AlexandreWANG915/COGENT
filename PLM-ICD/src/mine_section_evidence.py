#!/usr/bin/env python
"""Extract candidate codes with section-level attention for GPT verification.

For each sample:
1. Load sections from feather file
2. Run Longformer to get candidates and attention weights
3. Calculate attention distribution across sections for each code
4. Output candidates with section attention scores
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from longformer_multilabel_classifier import LongformerForMultilabelClassification


def get_section_char_ranges(text: str, sections_json: str) -> dict:
    """Get character ranges for each section in the text.

    Each section is searched independently from position 0, since sections
    in the JSON dict may not be in the same order as they appear in text.

    Returns:
        dict: {section_name: (char_start, char_end)}
    """
    if not sections_json:
        return {}

    try:
        sections = json.loads(sections_json)
        if not isinstance(sections, dict):
            return {}
    except (json.JSONDecodeError, TypeError):
        return {}

    section_ranges = {}
    text_lower = None  # Lazy initialization for case-insensitive search

    for section_name, section_text in sections.items():
        if not section_text or not section_text.strip():
            continue

        section_text_clean = section_text.strip()

        # Search for full section text from beginning
        idx = text.find(section_text_clean)

        if idx != -1:
            section_ranges[section_name] = (idx, idx + len(section_text_clean))
        else:
            # Try case-insensitive search (lazy init text_lower)
            if text_lower is None:
                text_lower = text.lower()
            section_lower = section_text_clean.lower()
            idx = text_lower.find(section_lower)
            if idx != -1:
                section_ranges[section_name] = (idx, idx + len(section_text_clean))

    return section_ranges


def calculate_section_attention(
    attention_weights: torch.Tensor,
    offset_mapping: list,
    section_ranges: dict,
    top_k_tokens: int = 100
) -> dict:
    """Calculate attention score for each section.

    Args:
        attention_weights: (seq_len,) attention weights for one label
        offset_mapping: List[(char_start, char_end)] from tokenizer
        section_ranges: {section_name: (char_start, char_end)}
        top_k_tokens: Only consider top K tokens with highest attention

    Returns:
        dict: {section_name: attention_score}
    """
    if not section_ranges:
        return {}

    att = attention_weights
    att_len = len(att)

    # Top-K filtering - use indices to handle ties properly
    if top_k_tokens > 0 and att_len > top_k_tokens:
        topk_indices = torch.topk(att, top_k_tokens).indices
        mask = torch.zeros(att_len, dtype=torch.bool)
        mask[topk_indices] = True
    else:
        mask = torch.ones(att_len, dtype=torch.bool)

    section_scores = {name: 0.0 for name in section_ranges.keys()}

    for token_idx, offset in enumerate(offset_mapping):
        if token_idx >= att_len:
            break

        if offset is None:
            continue

        char_start, char_end = offset
        if char_start == char_end:  # Skip special tokens
            continue

        if not mask[token_idx]:
            continue

        weight = att[token_idx].item()

        # Find which section this token belongs to
        for section_name, (sec_start, sec_end) in section_ranges.items():
            if char_start < sec_end and char_end > sec_start:
                section_scores[section_name] += weight
                break

    # Keep raw attention sums (not normalized) for cross-code comparison
    # Round to 4 decimal places and filter out zero scores
    section_scores = {k: round(v, 4) for k, v in section_scores.items() if v > 0}

    return section_scores


def main():
    parser = argparse.ArgumentParser(description="Extract candidates with section attention")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--feather_file", type=str, required=True, help="Path to feather file with sections")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation CSV file")
    parser.add_argument("--code_file", type=str, required=True, help="Path to ALL_CODES.txt")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--threshold", type=float, default=0.1, help="Probability threshold for candidates")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--top_k_tokens", type=int, default=100, help="Top K tokens for attention")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to process (0=all)")
    parser.add_argument("--split", type=str, default="val", help="Data split to use: 'train' or 'val'")
    args = parser.parse_args()

    # Load codes
    with open(args.code_file, 'r') as f:
        all_codes = [line.strip() for line in f if line.strip()]
    code_to_idx = {code: idx for idx, code in enumerate(all_codes)}
    idx_to_code = {idx: code for code, idx in code_to_idx.items()}
    num_labels = len(all_codes)
    print(f"Loaded {num_labels} codes")

    # Load feather file with sections
    print(f"Loading feather file: {args.feather_file}")
    df_feather = pd.read_feather(args.feather_file)
    print(f"  Total records: {len(df_feather)}")

    # Check if we should use feather directly or load separate val file
    if 'split' in df_feather.columns:
        # Use feather file directly with split column
        print(f"  Using 'split' column from feather file, split='{args.split}'")
        df_val = df_feather[df_feather['split'] == args.split].copy()
        df_val = df_val.reset_index(drop=True)

        # Ensure we have required columns
        if 'label' not in df_val.columns and 'target' in df_val.columns:
            # Convert target (array) to label (semicolon-separated string)
            df_val['label'] = df_val['target'].apply(lambda x: ';'.join(x) if isinstance(x, (list, np.ndarray)) else str(x))

        # Create note_id to sections mapping
        if 'sections_json' in df_feather.columns:
            note_to_sections = dict(zip(df_feather['note_id'], df_feather['sections_json'].fillna('')))
        else:
            note_to_sections = {nid: '' for nid in df_feather['note_id']}

        print(f"Loaded {len(df_val)} {args.split} samples from feather")
    else:
        # Fallback: Load separate val.csv file
        print("  No 'split' column found, loading from val_file")

        # Create note_id to sections mapping
        if 'sections_json' in df_feather.columns:
            note_to_sections = dict(zip(df_feather['note_id'], df_feather['sections_json'].fillna('')))
        else:
            note_to_sections = {nid: '' for nid in df_feather['note_id']}

        df_val = pd.read_csv(args.val_file)
        print(f"Loaded {len(df_val)} validation samples from CSV")

    if args.num_samples > 0:
        df_val = df_val.head(args.num_samples)
        print(f"  Using first {len(df_val)} samples")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LongformerForMultilabelClassification.from_pretrained(args.model_path)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()
    print(f"Model loaded: {args.model_path}")

    # Check LAAT mode
    base_model = model.module if hasattr(model, 'module') else model
    print(f"Model mode: {'LAAT' if base_model.use_laat else 'CLS'}")
    if not base_model.use_laat:
        raise ValueError("Model must be in LAAT mode to extract attention-based evidence")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Stats tracking
    all_preds = []
    all_labels = []
    stats = {
        'total_samples': len(df_val),
        'total_candidates': 0,
        'gt_candidates': 0,
        'non_gt_candidates': 0,
        'samples_with_sections': 0,
        'samples_without_sections': 0,
        'missing_note_ids': 0,
    }

    # Process in batches
    with open(args.output, 'w') as out_f:
        for batch_start in tqdm(range(0, len(df_val), args.batch_size), desc="Extracting"):
            batch_end = min(batch_start + args.batch_size, len(df_val))
            batch_df = df_val.iloc[batch_start:batch_end]

            texts = batch_df['text'].tolist()
            labels_strs = batch_df['label'].tolist()

            # Get note_ids if available
            if 'note_id' in batch_df.columns:
                note_ids = batch_df['note_id'].tolist()
            else:
                note_ids = [f"sample_{batch_start + i}" for i in range(len(batch_df))]

            # Parse ground truth labels
            batch_gt_codes = []
            batch_label_vecs = []
            for labels_str in labels_strs:
                gt_codes = set()
                label_vec = [0] * num_labels
                for code in labels_str.split(';'):
                    code = code.strip()
                    if code in code_to_idx:
                        gt_codes.add(code)
                        label_vec[code_to_idx[code]] = 1
                batch_gt_codes.append(gt_codes)
                batch_label_vecs.append(label_vec)

            # Tokenize
            encodings = tokenizer(
                texts,
                max_length=args.max_length,
                truncation=True,
                padding=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )

            # Forward pass
            with torch.no_grad():
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                global_attention_mask = torch.zeros_like(attention_mask)
                global_attention_mask[:, 0] = 1

                if hasattr(model, 'module'):
                    outputs, att_weights = model.module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        global_attention_mask=global_attention_mask,
                        return_laat_attention=True
                    )
                else:
                    outputs, att_weights = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        global_attention_mask=global_attention_mask,
                        return_laat_attention=True
                    )

                batch_probs = torch.sigmoid(outputs.logits).cpu().numpy()
                batch_att_weights = att_weights.cpu()

            # Collect predictions for metrics
            batch_preds = (batch_probs >= args.threshold).astype(int)
            all_preds.extend(batch_preds.tolist())
            all_labels.extend(batch_label_vecs)

            # Process each sample
            for i in range(len(batch_df)):
                sample_idx = batch_start + i
                text = texts[i]
                note_id = note_ids[i]
                gt_codes = batch_gt_codes[i]
                probs = batch_probs[i]
                sample_att_weights = batch_att_weights[i]
                offset_mapping = encodings['offset_mapping'][i].tolist()

                # Get sections for this note
                sections_json = note_to_sections.get(note_id)
                if sections_json is None:
                    stats['missing_note_ids'] += 1
                    sections_json = ''
                section_ranges = get_section_char_ranges(text, sections_json)

                if section_ranges:
                    stats['samples_with_sections'] += 1
                else:
                    stats['samples_without_sections'] += 1

                # Get candidates above threshold
                candidates = []
                for label_idx in range(num_labels):
                    prob = float(probs[label_idx])
                    if prob < args.threshold:
                        continue

                    code = idx_to_code[label_idx]
                    is_gt = code in gt_codes
                    attention = sample_att_weights[label_idx]

                    # Calculate section attention
                    section_attention = calculate_section_attention(
                        attention, offset_mapping, section_ranges,
                        top_k_tokens=args.top_k_tokens
                    )

                    candidates.append({
                        'code': code,
                        'prob': round(prob, 4),
                        'is_gt': is_gt,
                        'section_attention': section_attention,
                    })

                    stats['total_candidates'] += 1
                    if is_gt:
                        stats['gt_candidates'] += 1
                    else:
                        stats['non_gt_candidates'] += 1

                # Sort by probability
                candidates.sort(key=lambda x: -x['prob'])

                record = {
                    'sample_idx': sample_idx,
                    'note_id': note_id,
                    'gt_codes': list(gt_codes),
                    'num_gt': len(gt_codes),
                    'num_candidates': len(candidates),
                    'candidates': candidates,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    micro_prec = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_rec = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Extraction Summary (threshold={args.threshold})")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"  With sections: {stats['samples_with_sections']}")
    print(f"  Without sections: {stats['samples_without_sections']}")
    print(f"  Missing note_ids: {stats['missing_note_ids']}")
    print(f"Total candidates: {stats['total_candidates']}")
    print(f"  GT candidates: {stats['gt_candidates']}")
    print(f"  Non-GT candidates: {stats['non_gt_candidates']}")
    print(f"Avg candidates/sample: {stats['total_candidates']/stats['total_samples']:.2f}")
    print()
    print("Metrics:")
    print(f"  Micro - P: {micro_prec:.4f}, R: {micro_rec:.4f}, F1: {micro_f1:.4f}")
    print(f"  Macro - P: {macro_prec:.4f}, R: {macro_rec:.4f}, F1: {macro_f1:.4f}")
    print()
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
