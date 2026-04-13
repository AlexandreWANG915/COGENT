#!/usr/bin/env python3
"""
Run verifier inference over ICD code candidates.

The verifier reads structured evidence for each candidate, keeps supported
codes, and filters the rest.

Supports filtering by:
- code_prob_threshold: minimum probability to consider a candidate
- attention_threshold: minimum evidence score to keep a candidate-evidence link
"""

import os
os.environ['DISABLE_VERSION_CHECK'] = '1'

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import argparse
import json
import re
from tqdm import tqdm
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import time
import pandas as pd

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


# ICD-10 code descriptions used by the verifier
ICD10_DESCRIPTIONS = {
    "02HV33Z": "Insertion of Infusion Device into Superior Vena Cava, Percutaneous Approach",
    "D62.": "Acute posthemorrhagic anemia",
    "D64.9": "Anemia, unspecified",
    "D69.6": "Thrombocytopenia, unspecified",
    "E03.9": "Hypothyroidism, unspecified",
    "E11.22": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E66.9": "Obesity, unspecified",
    "E78.5": "Hyperlipidemia, unspecified",
    "E87.1": "Hypo-osmolality and hyponatremia",
    "E87.2": "Acidosis",
    "F17.210": "Nicotine dependence, cigarettes, uncomplicated",
    "F32.9": "Major depressive disorder, single episode, unspecified",
    "F41.9": "Anxiety disorder, unspecified",
    "G47.00": "Insomnia, unspecified",
    "G47.33": "Obstructive sleep apnea (adult) (pediatric)",
    "G89.29": "Other chronic pain",
    "I10.": "Essential (primary) hypertension",
    "I11.0": "Hypertensive heart disease with heart failure",
    "I12.9": "Hypertensive chronic kidney disease with stage 1-4 CKD",
    "I13.0": "Hypertensive heart and chronic kidney disease with heart failure",
    "I25.10": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
    "I25.2": "Old myocardial infarction",
    "I48.0": "Paroxysmal atrial fibrillation",
    "I48.91": "Unspecified atrial fibrillation",
    "J18.9": "Pneumonia, unspecified organism",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified",
    "J45.909": "Unspecified asthma, uncomplicated",
    "J96.01": "Acute respiratory failure with hypoxia",
    "K21.9": "Gastro-esophageal reflux disease without esophagitis",
    "K59.00": "Constipation, unspecified",
    "M10.9": "Gout, unspecified",
    "N17.9": "Acute kidney failure, unspecified",
    "N18.3": "Chronic kidney disease, stage 3 (moderate)",
    "N18.9": "Chronic kidney disease, unspecified",
    "N39.0": "Urinary tract infection, site not specified",
    "N40.0": "Benign prostatic hyperplasia without lower urinary tract symptoms",
    "Y92.230": "Patient room in hospital as the place of occurrence",
    "Y92.239": "Unspecified place in hospital as the place of occurrence",
    "Y92.9": "Unspecified place or not applicable",
    "Z23.": "Encounter for immunization",
    "Z66.": "Do not resuscitate",
    "Z79.01": "Long term (current) use of anticoagulants",
    "Z79.02": "Long term (current) use of antithrombotics/antiplatelets",
    "Z79.4": "Long term (current) use of insulin",
    "Z86.718": "Personal history of other venous thrombosis and embolism",
    "Z86.73": "Personal history of transient ischemic attack (TIA)",
    "Z87.891": "Personal history of nicotine dependence",
    "Z95.1": "Presence of aortocoronary bypass graft",
    "Z95.5": "Presence of coronary angioplasty implant and graft",
}


class Task2SectionVerifier:
    """Verify ICD codes using the verifier task format."""

    def __init__(self, model_path: str, template: str = "qwen",
                 num_gpus: int = 4, max_model_len: int = 8192,
                 adapter_path: str = None):
        """Initialize the verifier."""
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.adapter_path = adapter_path

        # Initialize tokenizer and template
        print(f"Loading tokenizer and template...")
        self.model_args, self.data_args, _, _ = get_infer_args(dict(
            model_name_or_path=model_path,
            template=template,
            cutoff_len=max_model_len,
            trust_remote_code=True,
        ))

        tokenizer_module = load_tokenizer(self.model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)

        if hasattr(self.template, 'mm_plugin') and self.template.mm_plugin:
            self.template.mm_plugin.expand_mm_tokens = False

        print("Tokenizer and template loaded")

        # Initialize vLLM
        print(f"Initializing vLLM with {num_gpus} GPU(s), max_model_len={max_model_len}...")
        llm_kwargs = dict(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",  # MedGemma requires bfloat16, float16 causes inf/nan
            tensor_parallel_size=num_gpus,
            disable_log_stats=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.85,
        )
        if adapter_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64
            print(f"LoRA adapter enabled: {adapter_path}")
        self.llm = LLM(**llm_kwargs)
        print(f"vLLM model loaded successfully, actual max_model_len={self.llm.llm_engine.model_config.max_model_len}")

        # Create LoRA request if adapter path is provided
        self.lora_request = LoRARequest("default", 1, adapter_path) if adapter_path else None

        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=4096,  # Longer for multiple codes
            stop_token_ids=self.template.get_stop_token_ids(self.tokenizer),
        )

    def build_prompt_single(self, section_name: str, section_content: str, code: str) -> str:
        """Build single-code verification prompt."""
        # No character-level truncation - let token-level check handle it

        desc = ICD10_DESCRIPTIONS.get(code, "")
        code_str = f"{code} - {desc}" if desc else code

        return f"""You are a medical coding specialist. Determine if the clinical evidence supports the ICD-10 diagnosis code.

Code: {code_str}

Section: {section_name}

Evidence:
{section_content}"""

    def parse_answer_single(self, text: str) -> bool:
        """Parse single-code verification answer."""
        pattern = r'<answer>\s*(Yes|No)\s*</answer>'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower() == 'yes'
        return False

    def build_prompt_multi_single(self, section_name: str, section_content: str, code: str) -> str:
        """Build multi-style prompt but for a single code."""
        desc = ICD10_DESCRIPTIONS.get(code, "")
        if desc:
            code_str = f"- {code} - {desc}"
        else:
            code_str = f"- {code} - "

        return f"""You are a medical coding specialist. Given the following section of a discharge summary, determine which ICD-10 codes are supported by the text.

Section: {section_name}

Content:
{section_content}

Candidate codes to verify:
{code_str}

For each code, determine if it is supported by THIS section's content:
- Answer "Yes" if there is:
  * Direct mention of the condition (e.g., "hypertension", "diabetes")
  * Abbreviations or synonyms (e.g., "HTN" = hypertension, "DM" = diabetes)
  * Historical references (e.g., "former smoker", "history of smoking" supports nicotine dependence history)
  * Related treatments or medications that imply the condition
  * Any reasonable clinical association
- Answer "No" ONLY if there is absolutely no evidence or relation to the diagnosis

For each code, output in the following format:
<code>CODE_HERE</code>
<think>Your reasoning about whether the evidence supports this code</think>
<answer>Yes or No</answer>"""

    def build_prompt(self, section_name: str, section_content: str, codes: List[str]) -> str:
        """Build verifier prompt."""
        # No character-level truncation - let token-level check in verify_batch handle it
        # If prompt is too long, it will be skipped with a warning

        # Build candidate codes list
        codes_list = []
        for code in codes:
            desc = ICD10_DESCRIPTIONS.get(code, "")
            if desc:
                codes_list.append(f"- {code} - {desc}")
            else:
                codes_list.append(f"- {code} - ")

        codes_str = "\n".join(codes_list)

        # Old prompt (stricter version):
        # For each code, determine if it is supported by THIS section's content:
        # - Answer "Yes" if there is direct mention, direct evidence, or reasonable clinical inference
        # - Answer "No" if there is no evidence or mention of the diagnosis

        return f"""You are a medical coding specialist. Given the following section of a discharge summary, determine which ICD-10 codes are supported by the text.

Section: {section_name}

Content:
{section_content}

Candidate codes to verify:
{codes_str}

For each code, determine if it is supported by THIS section's content:
- Answer "Yes" if there is:
  * Direct mention of the condition (e.g., "hypertension", "diabetes")
  * Abbreviations or synonyms (e.g., "HTN" = hypertension, "DM" = diabetes)
  * Historical references (e.g., "former smoker", "history of smoking" supports nicotine dependence history)
  * Related treatments or medications that imply the condition
  * Any reasonable clinical association
- Answer "No" ONLY if there is absolutely no evidence or relation to the diagnosis

For each code, output in the following format:
<code>CODE_HERE</code>
<think>Your reasoning about whether the evidence supports this code</think>
<answer>Yes or No</answer>"""

    def parse_answers(self, text: str, codes: List[str]) -> Dict[str, bool]:
        """Parse verification answers for multiple codes.

        Expected format:
        <code>CODE1</code>
        <think>reasoning...</think>
        <answer>Yes/No</answer>
        <code>CODE2</code>
        ...
        """
        results = {code: False for code in codes}

        # Find all code-answer pairs
        pattern = r'<code>\s*([^<]+?)\s*</code>.*?<answer>\s*(Yes|No)\s*</answer>'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for code_match, answer_match in matches:
            code = code_match.strip()
            verified = answer_match.lower() == 'yes'

            # Match code (handle potential variations)
            if code in results:
                results[code] = verified
            else:
                # Try to find closest match
                for orig_code in codes:
                    if code in orig_code or orig_code in code:
                        results[orig_code] = verified
                        break

        return results

    def verify_batch(self, items: List[Dict], mode: str = 'multi') -> List[Dict]:
        """Verify a batch of (section, codes) pairs.

        Args:
            items: List of dicts with 'section_name', 'section_content', 'codes'
            mode: 'single' for one code per prompt, 'multi' for multiple codes per prompt

        Returns:
            List of results with 'verified_codes' dict for each item
        """
        # Build prompts and encode
        all_inputs = []
        valid_indices = []

        for i, item in enumerate(items):
            section_name = item.get('section_name', '')
            section_content = item.get('section_content', '')
            codes = item.get('codes', [])

            if not section_content or len(section_content.strip()) < 10 or not codes:
                continue

            # Build prompt based on mode
            if mode == 'single':
                # Single mode: one code per prompt, simple format
                prompt = self.build_prompt_single(section_name, section_content, codes[0])
            elif mode == 'multi_single':
                # Multi-single mode: one code per prompt, multi-style format
                prompt = self.build_prompt_multi_single(section_name, section_content, codes[0])
            else:
                # Multi mode: multiple codes per prompt
                prompt = self.build_prompt(section_name, section_content, codes)

            conversations = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""}
            ]

            try:
                prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, conversations)
                # Reserve space for output (max_tokens from sampling_params)
                max_prompt_len = self.max_model_len - 4096  # Reserve 4096 for output
                if prompt_ids and len(prompt_ids) <= max_prompt_len:
                    all_inputs.append(prompt_ids)
                    valid_indices.append(i)
                elif prompt_ids:
                    print(f"Warning: Skipping item {i}, prompt too long: {len(prompt_ids)} > {max_prompt_len}")
            except Exception as e:
                print(f"Error encoding item {i}: {e}")
                continue

        if not all_inputs:
            return [{'verified_codes': {}, 'raw_output': ''} for _ in items]

        # Batch inference
        outputs = self.llm.generate(
            prompts=None,
            sampling_params=self.sampling_params,
            prompt_token_ids=all_inputs,
            lora_request=self.lora_request,
        )

        # Parse results
        results = [{'verified_codes': {code: False for code in item.get('codes', [])}, 'raw_output': ''} for item in items]

        for idx, output in zip(valid_indices, outputs):
            if output.outputs:
                text = output.outputs[0].text.strip()
                codes = items[idx].get('codes', [])

                if mode == 'single':
                    # Single mode: parse single answer
                    verified = self.parse_answer_single(text)
                    verified_codes = {codes[0]: verified}
                elif mode == 'multi_single':
                    # Multi-single mode: parse multi-style answer (but only one code)
                    verified_codes = self.parse_answers(text, codes)
                else:
                    # Multi mode: parse multiple answers
                    verified_codes = self.parse_answers(text, codes)

                results[idx] = {
                    'verified_codes': verified_codes,
                    'raw_output': text,
                }

        return results


def load_sections_from_feather(feather_path: str, note_ids: Set[str]) -> Dict[str, Dict[str, str]]:
    """Load sections for given note_ids from feather file.

    Returns:
        Dict mapping note_id -> {section_name: section_content}
    """
    print(f"Loading sections from {feather_path}...")
    df = pd.read_feather(feather_path)

    result = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing sections"):
        note_id = row['note_id']
        if note_id not in note_ids:
            continue

        sections_json = row.get('sections_json', '{}')
        try:
            sections = json.loads(sections_json) if isinstance(sections_json, str) else sections_json
        except json.JSONDecodeError:
            sections = {}

        if sections:
            result[note_id] = sections

    print(f"Loaded sections for {len(result)} notes")
    return result


# Define top-50 codes set for filtering
TOP50_CODES = set(ICD10_DESCRIPTIONS.keys())


def load_gt_codes_from_feather(feather_path: str, note_ids: Set[str]) -> Dict[str, Set[str]]:
    """Load complete gt_codes from feather file (filtered to top-50).

    The candidates file only contains gt_codes that were predicted by longformer
    with prob >= threshold. This function loads the COMPLETE gt_codes from the
    source feather file for accurate evaluation.

    Args:
        feather_path: Path to the feather file with original data
        note_ids: Set of note_ids to load

    Returns:
        Dict mapping note_id -> set of gt codes (only top-50 codes)
    """
    print(f"Loading complete gt_codes from {feather_path}...")
    df = pd.read_feather(feather_path)

    # Verify required columns exist
    if 'note_id' not in df.columns:
        raise ValueError(f"Missing 'note_id' column in {feather_path}")
    if 'target' not in df.columns:
        raise ValueError(f"Missing 'target' column in {feather_path}")

    result = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading gt_codes"):
        note_id = row['note_id']
        if note_id not in note_ids:
            continue

        target = row.get('target', [])

        # Handle various target types
        if target is None:
            target = []
        elif hasattr(target, 'tolist'):
            target = target.tolist()
        elif isinstance(target, str):
            # Handle JSON string case
            try:
                target = json.loads(target)
            except json.JSONDecodeError:
                target = []

        # Filter to top-50 codes only
        gt_codes = set(code for code in target if code in TOP50_CODES)
        result[note_id] = gt_codes

    print(f"Loaded complete gt_codes for {len(result)} notes")

    # Log coverage statistics
    missing = note_ids - set(result.keys())
    if missing:
        print(f"Warning: {len(missing)} note_ids not found in feather file")

    return result


def load_candidates(input_path: str, code_prob_threshold: float = 0.0,
                    attention_threshold: float = 0.0) -> Tuple[Dict[str, List[Dict]], Dict[str, Set[str]]]:
    """Load candidates and build section->codes mapping.

    Returns:
        note_candidates: Dict mapping note_id -> list of candidate dicts
        note_gt_codes: Dict mapping note_id -> set of ground truth codes
    """
    print(f"Loading candidates from {input_path}...")
    note_candidates = {}
    note_gt_codes = defaultdict(set)

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            note_id = data['note_id']
            gt_codes = set(data.get('gt_codes', []))
            note_gt_codes[note_id] = gt_codes

            # Filter candidates by prob threshold
            all_candidates = data.get('candidates', [])
            candidates = []
            for c in all_candidates:
                if c.get('prob', 0.0) >= code_prob_threshold:
                    # Filter section_attention by threshold
                    filtered_attention = {
                        sec: score for sec, score in c.get('section_attention', {}).items()
                        if score >= attention_threshold
                    }
                    if filtered_attention:
                        candidates.append({
                            'code': c['code'],
                            'prob': c['prob'],
                            'is_gt': c.get('is_gt', False),
                            'section_attention': filtered_attention,
                        })

            # If no candidates pass the filter, keep the best one (highest prob)
            if not candidates and all_candidates:
                best = max(all_candidates, key=lambda x: x.get('prob', 0.0))
                # Filter section_attention for the best candidate
                filtered_attention = {
                    sec: score for sec, score in best.get('section_attention', {}).items()
                    if score >= attention_threshold
                }
                # If even the best candidate has no sections after filtering,
                # keep the highest attention section
                if not filtered_attention and best.get('section_attention'):
                    top_section = max(best['section_attention'].items(), key=lambda x: x[1])
                    filtered_attention = {top_section[0]: top_section[1]}

                candidates = [{
                    'code': best['code'],
                    'prob': best['prob'],
                    'is_gt': best.get('is_gt', False),
                    'section_attention': filtered_attention,
                }]

            if candidates:
                note_candidates[note_id] = candidates

    print(f"Loaded {len(note_candidates)} notes with candidates")
    return note_candidates, dict(note_gt_codes)


def evaluate_predictions(samples: List[Dict]) -> Dict:
    """Calculate micro/macro F1, recall, precision."""
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import precision_recall_fscore_support

    all_codes = set()
    for sample in samples:
        gt = sample.get('gt_codes', [])
        pred = sample.get('predicted_codes', [])
        if not isinstance(gt, (list, set)):
            gt = []
        if not isinstance(pred, (list, set)):
            pred = []
        all_codes.update(gt)
        all_codes.update(pred)

    if not all_codes:
        return {
            'micro_precision': 0.0, 'micro_recall': 0.0, 'micro_f1': 0.0,
            'macro_precision': 0.0, 'macro_recall': 0.0, 'macro_f1': 0.0,
        }

    mlb = MultiLabelBinarizer(classes=sorted(all_codes))
    y_true = mlb.fit_transform([s.get('gt_codes', []) for s in samples])
    y_pred = mlb.transform([s.get('predicted_codes', []) for s in samples])

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)

    return {
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'micro_f1': micro_f1,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
    }


def save_results_to_csv(csv_path: str, args, stats: Dict, metrics: Dict, elapsed: float):
    """Append results to CSV file."""
    import csv
    from datetime import datetime

    # Create directory if needed
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    # Build result row
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': os.path.basename(args.model_path),
        'mode': args.mode,
        'code_prob_threshold': args.code_prob_threshold,
        'code_prob_upper': args.code_prob_upper,
        'attention_threshold': args.attention_threshold,
        'input_file': os.path.basename(args.input),
        'output_file': os.path.basename(args.output),
        'total_notes': stats['total_notes'],
        'total_items': stats['total_items'],
        'total_codes_verified': stats['total_codes_verified'],
        'notes_with_predictions': stats['notes_with_predictions'],
        'micro_precision': metrics['micro_precision'],
        'micro_recall': metrics['micro_recall'],
        'micro_f1': metrics['micro_f1'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'macro_f1': metrics['macro_f1'],
        'elapsed_seconds': elapsed,
        'items_per_sec': stats['total_items'] / elapsed if elapsed > 0 else 0.0,
    }

    fieldnames = list(row.keys())

    try:
        # Check if file exists and has content (to write header)
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except IOError as e:
        print(f"ERROR: Failed to write results to CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description='Verify ICD codes using the verifier task')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained SFT model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file (section_candidates)')
    parser.add_argument('--feather_file', type=str, required=True,
                        help='Feather file with section content')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file for predictions')
    parser.add_argument('--template', type=str, default='qwen',
                        help='Template name for LLaMA-Factory')
    parser.add_argument('--num_gpus', type=int, default=4,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--max_model_len', type=int, default=8192,
                        help='Maximum model context length')
    parser.add_argument('--code_prob_threshold', type=float, default=0.0,
                        help='Minimum code probability to consider a candidate')
    parser.add_argument('--code_prob_upper', type=float, default=1.0,
                        help='Upper threshold: codes with prob > this are kept without verification')
    parser.add_argument('--attention_threshold', type=float, default=0.0,
                        help='Minimum attention score to link candidate with section')
    parser.add_argument('--mode', type=str, default='multi', choices=['single', 'multi', 'multi_single'],
                        help='Verification mode: single, multi, or multi_single (one code per prompt with multi-style format)')
    parser.add_argument('--results_csv', type=str, default=None,
                        help='CSV file to append results (creates if not exists)')
    parser.add_argument('--adapter_path', type=str, default=None,
                        help='Path to LoRA adapter (optional, if not provided uses model_path directly)')
    parser.add_argument('--max_notes', type=int, default=None,
                        help='Maximum number of notes to process (for quick testing)')

    args = parser.parse_args()

    print(f"Code prob threshold: {args.code_prob_threshold}")
    print(f"Code prob upper: {args.code_prob_upper}")
    print(f"Attention threshold: {args.attention_threshold}")
    print(f"Verification mode: {args.mode}")

    # Load candidates
    note_candidates, note_gt_codes = load_candidates(
        args.input, args.code_prob_threshold, args.attention_threshold)

    # Get all note_ids
    note_ids = set(note_candidates.keys())

    # Limit notes if specified (for quick testing)
    if args.max_notes is not None and args.max_notes > 0 and args.max_notes < len(note_ids):
        sorted_note_ids = sorted(note_ids)[:args.max_notes]
        note_ids = set(sorted_note_ids)
        note_candidates = {k: v for k, v in note_candidates.items() if k in note_ids}
        note_gt_codes = {k: v for k, v in note_gt_codes.items() if k in note_ids}
        print(f"Limited to {args.max_notes} notes for testing")

    # Load sections from feather
    sections_by_note = load_sections_from_feather(args.feather_file, note_ids)

    # Load complete gt_codes from feather (for accurate evaluation)
    # The candidates file only has partial gt_codes (those predicted by longformer)
    complete_gt_codes = load_gt_codes_from_feather(args.feather_file, note_ids)

    # Initialize verifier
    verifier = Task2SectionVerifier(
        model_path=args.model_path,
        template=args.template,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        adapter_path=args.adapter_path,
    )

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # Build verification items based on mode
    print("\nPreparing verification items...")
    all_items = []
    # Track mapping: (note_id, section_name, code_or_none)
    # - For single mode: (note_id, section_name, code)
    # - For multi mode: (note_id, section_name, None)
    item_to_note_section = []

    # Track high-confidence codes that skip verification (prob > code_prob_upper)
    note_kept_codes = defaultdict(set)
    stats_kept_without_verify = 0
    stats_to_verify = 0

    for note_id, candidates in note_candidates.items():
        sections = sections_by_note.get(note_id, {})
        if not sections:
            continue

        # Separate candidates into: to_verify vs kept (high confidence)
        # Note: candidates already filtered by code_prob_threshold in load_candidates
        verify_candidates = []
        for c in candidates:
            if c['prob'] > args.code_prob_upper:
                # High confidence: keep without verification
                note_kept_codes[note_id].add(c['code'])
                stats_kept_without_verify += 1
            else:
                # Middle zone: needs verification
                verify_candidates.append(c)
                stats_to_verify += 1

        # Group codes by section (only for candidates needing verification)
        section_to_codes = defaultdict(list)
        for c in verify_candidates:
            code = c['code']
            for sec_name in c['section_attention'].keys():
                if sec_name in sections:
                    section_to_codes[sec_name].append(code)

        for sec_name, codes in section_to_codes.items():
            sec_content = sections.get(sec_name, '')
            if sec_content and codes:
                # Remove duplicates while preserving order
                unique_codes = list(dict.fromkeys(codes))

                if args.mode in ('single', 'multi_single'):
                    # Single/multi_single mode: create one item per (note_id, section, code)
                    for code in unique_codes:
                        all_items.append({
                            'section_name': sec_name,
                            'section_content': sec_content,
                            'codes': [code],  # Single code
                        })
                        item_to_note_section.append((note_id, sec_name, code))
                else:
                    # Multi mode: create one item per (note_id, section) with all codes
                    all_items.append({
                        'section_name': sec_name,
                        'section_content': sec_content,
                        'codes': unique_codes,  # Multiple codes
                    })
                    item_to_note_section.append((note_id, sec_name, None))

    print(f"Candidates kept without verification (prob > {args.code_prob_upper}): {stats_kept_without_verify}")
    print(f"Candidates to verify (prob <= {args.code_prob_upper}): {stats_to_verify}")
    print(f"Total verification items: {len(all_items)}")

    # Run verification (all at once)
    print("\nRunning verification...")
    all_results = verifier.verify_batch(all_items, mode=args.mode)

    # Aggregate results by note
    print("\nAggregating results...")
    note_verified_codes = defaultdict(set)
    debug_outputs = []  # For debugging

    for item_idx, result in enumerate(all_results):
        note_id, sec_name, _ = item_to_note_section[item_idx]
        verified_codes = result.get('verified_codes', {})
        raw_output = result.get('raw_output', '')

        # Save debug info for first 10 items
        if item_idx < 10:
            debug_outputs.append({
                'item_idx': item_idx,
                'note_id': note_id,
                'section_name': sec_name,
                'codes': all_items[item_idx].get('codes', []),
                'verified_codes': verified_codes,
                'raw_output': raw_output,
            })

        # Add all verified codes for this section
        for code, is_verified in verified_codes.items():
            if is_verified:
                note_verified_codes[note_id].add(code)

    # Write debug output
    debug_path = args.output.replace('.jsonl', '_debug.jsonl')
    print(f"\nWriting debug output to {debug_path}...")
    with open(debug_path, 'w') as f:
        for item in debug_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Build output records
    output_records = []
    stats = {
        'total_notes': len(note_candidates),
        'total_items': len(all_items),
        'total_codes_verified': sum(len(codes) for codes in note_verified_codes.values()),
        'notes_with_predictions': 0,
    }

    for note_id in note_candidates.keys():
        # Combine: high-confidence codes (kept without verification) + verified codes
        kept = note_kept_codes.get(note_id, set())
        verified = note_verified_codes.get(note_id, set())
        all_predicted = kept | verified
        predicted_codes = sorted(all_predicted)
        # Use COMPLETE gt_codes from feather file (not the partial ones from candidates)
        gt_codes = sorted(complete_gt_codes.get(note_id, set()))
        candidate_codes = [c['code'] for c in note_candidates[note_id]]

        output_record = {
            'note_id': note_id,
            'gt_codes': gt_codes,
            'candidate_codes': candidate_codes,
            'predicted_codes': predicted_codes,
            'kept_codes': sorted(kept),  # For debugging
            'verified_codes': sorted(verified),  # For debugging
        }
        output_records.append(output_record)

        if predicted_codes:
            stats['notes_with_predictions'] += 1

    # Write output
    print(f"\nWriting results to {args.output}...")
    with open(args.output, 'w') as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print(f"Code prob threshold: {args.code_prob_threshold}")
    print(f"Code prob upper: {args.code_prob_upper}")
    print(f"Attention threshold: {args.attention_threshold}")
    print(f"Total notes: {stats['total_notes']}")
    print(f"Total section items: {stats['total_items']}")
    print(f"Total codes verified: {stats['total_codes_verified']}")
    print(f"Notes with predictions: {stats['notes_with_predictions']}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Rate: {stats['total_items']/max(elapsed, 0.001):.1f} items/sec")

    # Run evaluation
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    metrics = evaluate_predictions(output_records)
    print(f"Micro Precision: {metrics['micro_precision']:.4f}")
    print(f"Micro Recall:    {metrics['micro_recall']:.4f}")
    print(f"Micro F1:        {metrics['micro_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")

    # Save to CSV if specified
    if args.results_csv:
        save_results_to_csv(args.results_csv, args, stats, metrics, elapsed)
        print(f"\nResults appended to: {args.results_csv}")

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
