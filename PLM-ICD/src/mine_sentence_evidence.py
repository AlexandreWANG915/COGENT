#!/usr/bin/env python
"""Extract candidate codes with sentence-level attention for LLM verification.

For each sample:
1. Load sections from feather file
2. Split sections into sentences (fallback to whole section/text if splitting fails)
3. Run Longformer to get candidates and attention weights
4. Calculate attention distribution across sentences for each code
5. Output candidates with sentence attention scores

Key difference from extract_evidence_v2.py:
- Every sample gets output (no skipping)
- If sentence splitting fails, use whole section/text as fallback
- Output format similar to section_candidates for easy integration
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
from evidence_extraction.sentence_splitter import split_sections_into_sentences, split_text_into_sentences


def get_sentence_char_ranges(sentences: list) -> dict:
    """Convert sentence list to {sentence_id: (char_start, char_end)} format."""
    ranges = {}
    for i, sent in enumerate(sentences):
        sent_id = f"sent_{i}"
        ranges[sent_id] = (sent['char_start'], sent['char_end'])
    return ranges


def create_fallback_sentences(text: str, sections_json: str) -> list:
    """Create fallback sentences when normal splitting fails.

    Strategy:
    1. Try to use whole sections as "sentences"
    2. If no sections, use whole text as one "sentence"
    """
    sentences = []

    # Try to parse sections
    try:
        sections = json.loads(sections_json) if sections_json else {}
    except (json.JSONDecodeError, TypeError):
        sections = {}

    if sections:
        # Use each section as a "sentence"
        # Track cumulative position to avoid overlapping ranges
        char_pos = 0
        for section_name, section_text in sections.items():
            if section_text and section_text.strip():
                section_text = section_text.strip()
                # Try to find actual position in text
                start = text.find(section_text[:min(100, len(section_text))])
                if start == -1:
                    # Use cumulative position if not found
                    start = char_pos
                sentences.append({
                    'text': section_text,
                    'section': section_name,
                    'char_start': start,
                    'char_end': start + len(section_text)
                })
                # Update cumulative position
                char_pos = max(char_pos, start + len(section_text) + 1)

    if not sentences:
        # Use whole text as one "sentence"
        sentences.append({
            'text': text.strip() if text else "",
            'section': 'full_text',
            'char_start': 0,
            'char_end': len(text) if text else 0
        })

    return sentences


def calculate_sentence_attention(
    attention_weights: torch.Tensor,
    offset_mapping: list,
    sentence_ranges: dict,
    top_k_tokens: int = 100
) -> dict:
    """Calculate attention score for each sentence.

    Args:
        attention_weights: (seq_len,) attention weights for one label
        offset_mapping: List[(char_start, char_end)] from tokenizer
        sentence_ranges: {sentence_id: (char_start, char_end)}
        top_k_tokens: Only consider top K tokens with highest attention

    Returns:
        dict: {sentence_id: attention_score}
    """
    if not sentence_ranges:
        return {}

    att = attention_weights
    att_len = len(att)

    # Top-K filtering
    if top_k_tokens > 0 and att_len > top_k_tokens:
        topk_indices = torch.topk(att, top_k_tokens).indices
        mask = torch.zeros(att_len, dtype=torch.bool)
        mask[topk_indices] = True
    else:
        mask = torch.ones(att_len, dtype=torch.bool)

    sentence_scores = {name: 0.0 for name in sentence_ranges.keys()}

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

        # Find which sentence this token belongs to
        for sent_id, (sent_start, sent_end) in sentence_ranges.items():
            if char_start < sent_end and char_end > sent_start:
                sentence_scores[sent_id] += weight
                break

    # Round and filter zero scores
    sentence_scores = {k: round(v, 4) for k, v in sentence_scores.items() if v > 0}

    return sentence_scores


def main():
    parser = argparse.ArgumentParser(description="Extract candidates with sentence attention")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--feather_file", type=str, required=True, help="Path to feather file with sections")
    parser.add_argument("--code_file", type=str, required=True, help="Path to ALL_CODES.txt")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--threshold", type=float, default=0.1, help="Probability threshold for candidates")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--top_k_tokens", type=int, default=100, help="Top K tokens for attention")
    parser.add_argument("--top_k_sentences", type=int, default=5, help="Top K sentences to keep per code")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to process (0=all)")
    parser.add_argument("--split", type=str, default="val", help="Data split to use: 'train', 'val', or 'test'")
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

    # Filter by split
    if 'split' in df_feather.columns:
        print(f"  Using 'split' column, split='{args.split}'")
        df_val = df_feather[df_feather['split'] == args.split].copy()
        df_val = df_val.reset_index(drop=True)

        if 'label' not in df_val.columns and 'target' in df_val.columns:
            df_val['label'] = df_val['target'].apply(
                lambda x: ';'.join(x) if isinstance(x, (list, np.ndarray)) else str(x)
            )
        print(f"Loaded {len(df_val)} {args.split} samples from feather")
    else:
        raise ValueError("Feather file must have 'split' column")

    if args.num_samples > 0:
        df_val = df_val.head(args.num_samples)
        print(f"  Using first {len(df_val)} samples")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LongformerForMultilabelClassification.from_pretrained(args.model_path)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()
    print(f"Model loaded: {args.model_path}")

    base_model = model.module if hasattr(model, 'module') else model
    print(f"Model mode: {'LAAT' if base_model.use_laat else 'CLS'}")
    if not base_model.use_laat:
        raise ValueError("Model must be in LAAT mode to extract attention-based evidence")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Stats tracking
    stats = {
        'total_samples': len(df_val),
        'total_candidates': 0,
        'gt_candidates': 0,
        'non_gt_candidates': 0,
        'samples_with_sentences': 0,
        'samples_with_fallback': 0,
        'samples_no_gt': 0,
    }

    # For metrics calculation
    all_preds = []
    all_labels = []

    with open(args.output, 'w') as out_f:
        for batch_start in tqdm(range(0, len(df_val), args.batch_size), desc="Extracting"):
            batch_end = min(batch_start + args.batch_size, len(df_val))
            batch_df = df_val.iloc[batch_start:batch_end]

            texts = batch_df['text'].tolist()
            labels_strs = batch_df['label'].tolist()
            note_ids = batch_df['note_id'].tolist() if 'note_id' in batch_df.columns else [f"sample_{batch_start + i}" for i in range(len(batch_df))]
            sections_jsons = batch_df['sections_json'].fillna('{}').tolist() if 'sections_json' in batch_df.columns else ['{}'] * len(batch_df)

            # Parse ground truth labels
            batch_gt_codes = []
            batch_label_vecs = []
            for labels_str in labels_strs:
                gt_codes = set()
                label_vec = [0] * num_labels
                for code in str(labels_str).split(';'):
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
                text = texts[i]
                note_id = note_ids[i]
                sections_json = sections_jsons[i]
                gt_codes = batch_gt_codes[i]
                probs = batch_probs[i]
                sample_att_weights = batch_att_weights[i]
                offset_mapping = encodings['offset_mapping'][i].tolist()

                # Split into sentences
                sentences = split_sections_into_sentences(sections_json, text)
                if not sentences:
                    sentences = split_text_into_sentences(text)

                # Fallback if still no sentences
                used_fallback = False
                if not sentences:
                    sentences = create_fallback_sentences(text, sections_json)
                    used_fallback = True
                    stats['samples_with_fallback'] += 1
                else:
                    stats['samples_with_sentences'] += 1

                # Build sentence_id -> info mapping
                sentence_info = {f"sent_{j}": sent for j, sent in enumerate(sentences)}
                sentence_ranges = get_sentence_char_ranges(sentences)

                # Track if sample has gt
                if not gt_codes:
                    stats['samples_no_gt'] += 1

                # Get candidates above threshold
                candidates = []
                for label_idx in range(num_labels):
                    prob = float(probs[label_idx])
                    if prob < args.threshold:
                        continue

                    code = idx_to_code[label_idx]
                    is_gt = code in gt_codes
                    attention = sample_att_weights[label_idx]

                    # Calculate sentence attention
                    sentence_attention = calculate_sentence_attention(
                        attention, offset_mapping, sentence_ranges,
                        top_k_tokens=args.top_k_tokens
                    )

                    # Get top-k sentences with their text and section
                    top_sentences = []
                    sorted_sents = sorted(sentence_attention.items(), key=lambda x: -x[1])
                    for sent_id, score in sorted_sents[:args.top_k_sentences]:
                        sent_info = sentence_info.get(sent_id, {})
                        top_sentences.append({
                            'text': sent_info.get('text', ''),
                            'section': sent_info.get('section', 'unknown'),
                            'score': score
                        })

                    candidates.append({
                        'code': code,
                        'prob': round(prob, 4),
                        'is_gt': is_gt,
                        'sentence_attention': sentence_attention,
                        'top_sentences': top_sentences,
                    })

                    stats['total_candidates'] += 1
                    if is_gt:
                        stats['gt_candidates'] += 1
                    else:
                        stats['non_gt_candidates'] += 1

                # Sort by probability
                candidates.sort(key=lambda x: -x['prob'])

                record = {
                    'sample_idx': batch_start + i,
                    'note_id': note_id,
                    'gt_codes': list(gt_codes),
                    'num_gt': len(gt_codes),
                    'num_candidates': len(candidates),
                    'num_sentences': len(sentences),
                    'used_fallback': used_fallback,
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
    print(f"  With proper sentences: {stats['samples_with_sentences']}")
    print(f"  With fallback (section/text): {stats['samples_with_fallback']}")
    print(f"  Without GT codes: {stats['samples_no_gt']}")
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
