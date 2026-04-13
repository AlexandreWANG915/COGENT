#!/usr/bin/env python3
"""
GPT verify section-level seed data for Qwen LLM training.

Process:
1. Load seed data samples (each has one code + one section)
2. Batch samples by section to reduce API calls
3. GPT verifies all codes for each section
4. Filter to keep consistent predictions:
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
from typing import Dict, List
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


def build_batch_prompt(section_name: str, section_content: str,
                       codes_with_desc: List[Dict]) -> str:
    """Build verification prompt for a batch of codes in one section."""
    # Build candidate list
    candidate_lines = []
    for item in codes_with_desc:
        code = item['code']
        desc = item['code_description']
        candidate_lines.append(f"- {code}: {desc}")

    candidate_text = "\n".join(candidate_lines)

    return f"""You are a medical coding specialist. Given the following section of a discharge summary, determine which ICD-10 codes are supported by the text.

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
<answer>Yes or No</answer>

Example output:
<code>I10</code>
<think>The section mentions the patient has a history of hypertension and is currently on lisinopril for blood pressure management. Blood pressure readings of 140/90 are documented during the hospital stay.</think>
<answer>Yes</answer>

<code>E11.9</code>
<think>There is no mention of diabetes, glucose levels, HbA1c, or diabetic medications in this section. The patient's metabolic panel does not indicate any glucose abnormalities.</think>
<answer>No</answer>

Provide output for each code in the candidate list."""


def parse_tag_response(response: str) -> Dict:
    """Parse tag-based response from GPT.

    Expected format:
    <code>I10</code>
    <think>reasoning...</think>
    <answer>Yes</answer>
    ...
    """
    results = {}

    # Find all code blocks
    code_pattern = r'<code>(.*?)</code>'
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'

    codes = re.findall(code_pattern, response, re.DOTALL)
    thinks = re.findall(think_pattern, response, re.DOTALL)
    answers = re.findall(answer_pattern, response, re.DOTALL)

    # Match by position
    for i, code in enumerate(codes):
        code = code.strip()
        think = thinks[i].strip() if i < len(thinks) else ""
        answer = answers[i].strip() if i < len(answers) else ""

        # Normalize answer
        answer_lower = answer.lower()
        if 'yes' in answer_lower:
            valid = True
        elif 'no' in answer_lower:
            valid = False
        else:
            valid = None

        results[code] = {
            'valid': valid,
            'think': think,
            'answer': answer
        }

    return results


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
                max_tokens=4000
            )
            # Track tokens
            if response.usage:
                token_tracker.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise e


def verify_batch(args) -> Dict:
    """Verify a batch of codes for a single section."""
    client, batch_key, section_name, section_content, codes_with_desc, sample_indices = args

    if not section_content or not section_content.strip():
        return {
            'batch_key': batch_key,
            'results': {},
            'sample_indices': sample_indices,
            'status': 'empty_section'
        }

    prompt = build_batch_prompt(section_name, section_content, codes_with_desc)

    try:
        response = call_gpt(client, prompt)
        parsed = parse_tag_response(response)

        return {
            'batch_key': batch_key,
            'results': parsed,
            'sample_indices': sample_indices,
            'raw_response': response,
            'status': 'success'
        }
    except Exception as e:
        return {
            'batch_key': batch_key,
            'results': {},
            'sample_indices': sample_indices,
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="GPT verify seed data")
    parser.add_argument("--input", type=str, default="data_v4/seed_data/section_seed_samples.jsonl",
                        help="Input seed samples JSONL")
    parser.add_argument("--output", type=str, default="data_v4/seed_data/section_seed_verified.jsonl",
                        help="Output verified results JSONL")
    parser.add_argument("--final", type=str, default="data_v4/seed_data/section_seed_final.jsonl",
                        help="Final filtered seed data")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    parser.add_argument("--batch_size", type=int, default=20, help="Max codes per batch")
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

    # Group samples by (note_id, section_name) for batch processing
    # Each batch contains up to batch_size codes for the same section
    print("\nGrouping samples for batch processing...")
    batch_groups = defaultdict(list)  # (note_id, section_name) -> list of (sample_idx, sample)

    for idx, sample in enumerate(samples):
        if idx in verified_indices:
            continue
        key = (sample['note_id'], sample['section_name'])
        batch_groups[key].append((idx, sample))

    print(f"  {len(batch_groups)} unique (note_id, section) combinations")

    # Create batches (split large groups)
    batches = []
    for key, items in batch_groups.items():
        note_id, section_name = key
        # Get section content from first sample
        section_content = items[0][1]['section_content']

        # Split into batches of batch_size
        for i in range(0, len(items), args.batch_size):
            batch_items = items[i:i + args.batch_size]
            sample_indices = [item[0] for item in batch_items]
            codes_with_desc = [
                {'code': item[1]['code'], 'code_description': item[1]['code_description']}
                for item in batch_items
            ]

            batch_key = f"{note_id}_{section_name}_{i}"
            batches.append((batch_key, section_name, section_content, codes_with_desc, sample_indices))

    print(f"  Created {len(batches)} batches to process")

    # Create client
    client = create_client()

    # Build tasks
    tasks = []
    for batch in batches:
        batch_key, section_name, section_content, codes_with_desc, sample_indices = batch
        tasks.append((client, batch_key, section_name, section_content, codes_with_desc, sample_indices))

    print(f"\nVerifying with {args.workers} workers...")

    # Process batches
    batch_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(verify_batch, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks)):
            result = future.result()
            batch_results.append(result)

    # Map batch results back to samples
    sample_verdicts = {}  # sample_idx -> verdict
    for batch_result in batch_results:
        sample_indices = batch_result['sample_indices']
        gpt_results = batch_result['results']
        status = batch_result['status']

        # Map each sample index to its verdict
        for idx in sample_indices:
            sample = samples[idx]
            code = sample['code']

            if status != 'success':
                sample_verdicts[idx] = {'status': status, 'valid': None, 'think': '', 'answer': 'API error'}
            elif code in gpt_results:
                verdict = gpt_results[code]
                sample_verdicts[idx] = {
                    'status': 'success',
                    'valid': verdict.get('valid'),
                    'think': verdict.get('think', ''),
                    'answer': verdict.get('answer', '')
                }
            else:
                sample_verdicts[idx] = {'status': 'not_found', 'valid': None, 'think': '', 'answer': 'Code not in GPT response'}

    # Merge with checkpoint results
    for idx, record in checkpoint_results.items():
        if idx not in sample_verdicts:
            sample_verdicts[idx] = {
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
            'section_name': sample['section_name'],
            'section_score': sample['section_score'],
            'section_content': sample['section_content'],
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
            'api_calls': len(batch_results),
            'final_samples': len(final_samples),
            **cost_summary
        }, f, indent=2)
    print(f"Saved cost summary to {cost_file}")

    print(f"\nDone! Final seed data: {len(final_samples)} samples")


if __name__ == "__main__":
    main()
