#!/usr/bin/env python3
"""
GPT verify sentence-level seed data for Qwen LLM training.

Process:
1. Load sentence seed samples (each has one code + one sentence)
2. GPT verifies if sentence supports the code
3. Filter to keep consistent predictions:
   - GT (is_gt=True) + GPT says Yes → keep as positive
   - HN (is_gt=False) + GPT says No → keep as negative
"""

import json
import re
import time
import argparse
import os
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Tuple
from collections import defaultdict

# Azure OpenAI configuration is provided via environment variables.
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o")


def create_client():
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not api_key or not azure_endpoint:
        raise RuntimeError("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT before running this script.")

    return AzureOpenAI(
        api_key=api_key,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=azure_endpoint
    )


def build_prompt(code: str, description: str, sentence: str, section: str) -> str:
    """Build verification prompt for a single sentence."""
    return f"""You are a medical coding specialist. Determine if the clinical evidence supports the ICD-10 diagnosis code.

Code: {code} - {description}

Section: {section}
Evidence: {sentence}

Determine if this evidence supports the diagnosis code:
- Answer "Yes" if there is direct mention, direct evidence, or reasonable clinical inference
- Answer "No" if there is no evidence or mention of the diagnosis

Output format:
<think>Your reasoning about whether the evidence supports the code</think>
<answer>Yes or No</answer>"""


def parse_response(response: str) -> Tuple[str, str, bool]:
    """Parse <think> and <answer> from response."""
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    # Normalize answer
    answer_lower = answer.lower()
    if 'yes' in answer_lower:
        valid = True
    elif 'no' in answer_lower:
        valid = False
    else:
        valid = None

    return think, answer, valid


class TokenTracker:
    """Track token usage and cost."""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # GPT-4o pricing (per 1M tokens)
        self.input_price = 2.50
        self.output_price = 10.00

    def add(self, input_tokens, output_tokens):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_cost(self):
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_price
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_price
        return input_cost + output_cost

    def summary(self):
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost_usd': round((self.total_input_tokens / 1_000_000) * self.input_price, 4),
            'output_cost_usd': round((self.total_output_tokens / 1_000_000) * self.output_price, 4),
            'total_cost_usd': round(self.get_cost(), 4)
        }


# Global token tracker
token_tracker = TokenTracker()


def call_gpt(client, prompt: str, max_retries: int = 5) -> str:
    """Call GPT with exponential backoff retry."""
    global token_tracker
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            # Track tokens
            if response.usage:
                token_tracker.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 0.5 * (attempt + 1)  # Shorter retry: 0.5, 1.0, 1.5, 2.0, 2.5 seconds
                time.sleep(wait_time)
            else:
                raise e


def verify_sample(args) -> Dict:
    """Verify a single sample."""
    client, sample_idx, sample = args

    code = sample['code']
    description = sample['code_description']
    sentence = sample['sentence_text']
    section = sample.get('sentence_section', '')

    prompt = build_prompt(code, description, sentence, section)

    try:
        response = call_gpt(client, prompt)
        think, answer, valid = parse_response(response)

        return {
            'sample_idx': sample_idx,
            'status': 'success',
            'valid': valid,
            'think': think,
            'answer': answer
        }
    except Exception as e:
        return {
            'sample_idx': sample_idx,
            'status': 'error',
            'valid': None,
            'think': '',
            'answer': '',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="GPT verify sentence seed data")
    parser.add_argument("--input", type=str, default="data_v4/seed_data/sentence_seed_samples.jsonl",
                        help="Input seed samples JSONL")
    parser.add_argument("--output", type=str, default="data_v4/seed_data/sentence_seed_verified.jsonl",
                        help="Output verified results JSONL")
    parser.add_argument("--final", type=str, default="data_v4/seed_data/sentence_seed_final.jsonl",
                        help="Final filtered seed data")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint file")
    args = parser.parse_args()

    # Load samples
    print(f"Loading samples from {args.input}...")
    samples = []
    with open(args.input) as f:
        for line in f:
            samples.append(json.loads(line))

    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"  Loaded {len(samples)} samples")

    # Resume from checkpoint if provided
    verified_indices = set()
    checkpoint_results = {}
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        with open(args.resume) as f:
            for line in f:
                record = json.loads(line)
                idx = record.get('sample_idx')
                if idx is not None:
                    verified_indices.add(idx)
                    checkpoint_results[idx] = record
        print(f"  Loaded {len(verified_indices)} already verified samples")

    # Build tasks for unverified samples
    client = create_client()
    tasks = []
    for idx, sample in enumerate(samples):
        if idx not in verified_indices:
            tasks.append((client, idx, sample))

    print(f"\nVerifying {len(tasks)} samples with {args.workers} workers...")

    # Process with parallel workers
    sample_verdicts = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(verify_sample, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks)):
            result = future.result()
            idx = result['sample_idx']
            sample_verdicts[idx] = result

    # Merge with checkpoint results
    for idx, record in checkpoint_results.items():
        if idx not in sample_verdicts:
            sample_verdicts[idx] = {
                'sample_idx': idx,
                'status': 'success',
                'valid': record.get('gpt_valid'),
                'think': record.get('gpt_think', ''),
                'answer': record.get('gpt_answer', '')
            }

    # Save all verified results
    print(f"\nSaving verified results to {args.output}...")
    verified_samples = []
    for idx, sample in enumerate(samples):
        verdict = sample_verdicts.get(idx, {'status': 'not_processed', 'valid': None, 'think': '', 'answer': ''})

        verified_sample = {
            'sample_idx': idx,
            'note_id': sample['note_id'],
            'code': sample['code'],
            'code_description': sample['code_description'],
            'is_gt': sample['is_gt'],
            'prob': sample['prob'],
            'sentence_text': sample['sentence_text'],
            'sentence_section': sample.get('sentence_section', ''),
            'sentence_score': sample.get('sentence_score', 0),
            'gpt_status': verdict['status'],
            'gpt_valid': verdict['valid'],
            'gpt_think': verdict['think'],
            'gpt_answer': verdict['answer']
        }
        verified_samples.append(verified_sample)

    with open(args.output, 'w') as f:
        for sample in verified_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Filter to keep consistent predictions
    print(f"\nFiltering consistent predictions...")
    final_samples = []
    stats = defaultdict(int)

    for sample in verified_samples:
        is_gt = sample['is_gt']
        gpt_valid = sample['gpt_valid']

        if gpt_valid is None:
            stats['skipped_no_verdict'] += 1
            continue

        # GT + GPT Yes → keep as positive
        if is_gt and gpt_valid:
            sample['final_label'] = 'positive'
            final_samples.append(sample)
            stats['positive'] += 1

        # HN + GPT No → keep as negative
        elif not is_gt and not gpt_valid:
            sample['final_label'] = 'negative'
            final_samples.append(sample)
            stats['negative'] += 1

        # GT + GPT No → discard (disagreement)
        elif is_gt and not gpt_valid:
            stats['discarded_gt_no'] += 1

        # HN + GPT Yes → discard (disagreement)
        elif not is_gt and gpt_valid:
            stats['discarded_hn_yes'] += 1

    print(f"\nStatistics:")
    print(f"  Total samples: {len(verified_samples)}")
    print(f"  Kept positive (GT + Yes): {stats['positive']}")
    print(f"  Kept negative (HN + No): {stats['negative']}")
    print(f"  Discarded (GT + No): {stats['discarded_gt_no']}")
    print(f"  Discarded (HN + Yes): {stats['discarded_hn_yes']}")
    print(f"  Skipped (no verdict): {stats['skipped_no_verdict']}")
    print(f"  Final samples: {len(final_samples)}")

    # Per-code statistics
    code_stats = defaultdict(lambda: {'positive': 0, 'negative': 0})
    for sample in final_samples:
        code = sample['code']
        if sample['final_label'] == 'positive':
            code_stats[code]['positive'] += 1
        else:
            code_stats[code]['negative'] += 1

    print(f"\nPer-code statistics:")
    for code in sorted(code_stats.keys()):
        s = code_stats[code]
        print(f"  {code}: pos={s['positive']}, neg={s['negative']}")

    # Save final filtered samples
    print(f"\nSaving final seed data to {args.final}...")
    with open(args.final, 'w') as f:
        for sample in final_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Print and save cost summary
    cost_summary = token_tracker.summary()
    print(f"\n{'='*50}")
    print("API COST SUMMARY")
    print(f"{'='*50}")
    print(f"  Input tokens:  {cost_summary['input_tokens']:,}")
    print(f"  Output tokens: {cost_summary['output_tokens']:,}")
    print(f"  Total tokens:  {cost_summary['total_tokens']:,}")
    print(f"  Input cost:    ${cost_summary['input_cost_usd']:.4f}")
    print(f"  Output cost:   ${cost_summary['output_cost_usd']:.4f}")
    print(f"  TOTAL COST:    ${cost_summary['total_cost_usd']:.4f}")
    print(f"{'='*50}")

    # Save cost summary
    cost_file = args.final.replace('.jsonl', '_cost.json')
    with open(cost_file, 'w') as f:
        json.dump({
            'samples_processed': len(verified_samples),
            'final_samples': len(final_samples),
            **cost_summary
        }, f, indent=2)
    print(f"Saved cost summary to {cost_file}")

    print(f"\nDone! Final seed data: {len(final_samples)} samples")


if __name__ == "__main__":
    main()
