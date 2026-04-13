# coding=utf-8
"""
Training script for Clinical-Longformer on ICD coding task.
Supports LAAT/CLS modes and pretrained/scratch training.
"""

import argparse
import logging
import math
import os
import random

import datasets
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
import torch
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import (
    AutoConfig,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)
from torch.optim import AdamW

from longformer_multilabel_classifier import LongformerForMultilabelClassification
from icd_metrics import all_metrics, micro_f1

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Clinical-Longformer for ICD coding")

    # Data arguments
    parser.add_argument("--train_file", type=str, required=True, help="Training data JSON file")
    parser.add_argument("--validation_file", type=str, required=True, help="Validation data JSON file")
    parser.add_argument("--code_file", type=str, required=True, help="File containing all ICD codes")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="yikuan8/Clinical-Longformer",
                        help="Pretrained model name or path")
    parser.add_argument("--use_laat", action="store_true", help="Use LAAT attention mechanism")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch without pretrained weights")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length (up to 4096)")

    # Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine", "constant"])
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")
    parser.add_argument("--save_every_epoch", action="store_true",
                        help="Save checkpoint after every epoch (overwrites previous)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="plm-icd", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Early stopping patience (0 to disable)")

    args = parser.parse_args()

    # Validate max_length
    args.max_length = min(args.max_length, 4096)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision="bf16")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Initialize wandb
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("WandB requested but not installed. Disabling.")
            args.use_wandb = False
        elif accelerator.is_main_process:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load ICD codes
    labels = []
    with open(args.code_file, "r") as f:
        for line in f:
            if line.strip():
                labels.append(line.strip())
    label_list = sorted(labels)
    num_labels = len(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}

    logger.info(f"Number of labels: {num_labels}")

    # Load config and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = num_labels
    config.use_laat = args.use_laat

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load model
    if args.from_scratch:
        logger.info("Training from scratch (random initialization)")
        model = LongformerForMultilabelClassification(config)
    else:
        logger.info("Loading pretrained weights")
        model = LongformerForMultilabelClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        )

    mode_str = "LAAT" if args.use_laat else "CLS"
    init_str = "scratch" if args.from_scratch else "pretrained"
    logger.info(f"Model mode: {mode_str}, Init: {init_str}")

    # Load datasets - auto detect format (csv or json)
    data_files = {"train": args.train_file, "validation": args.validation_file}
    file_ext = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(file_ext, data_files=data_files)

    # Preprocess function
    def preprocess_function(examples):
        result = tokenizer(
            examples["text"],
            padding=False,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=True,
        )

        # Process labels
        if "label" in examples:
            result["label_ids"] = [
                [label_to_id[label.strip()] for label in labels_str.strip().split(';') if label.strip()]
                if labels_str else []
                for labels_str in examples["label"]
            ]

        return result

    # Process datasets
    remove_columns = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # Data collator - NO CHUNKING, direct 2D format
    def data_collator(features):
        batch = {}

        # Find max length in batch (up to max_length)
        max_len = min(max(len(f["input_ids"]) for f in features), args.max_length)

        # Pad input_ids - 2D format: (batch_size, seq_len)
        batch["input_ids"] = torch.tensor([
            f["input_ids"][:max_len] + [tokenizer.pad_token_id] * (max_len - len(f["input_ids"][:max_len]))
            for f in features
        ])

        # Pad attention_mask
        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.tensor([
                f["attention_mask"][:max_len] + [0] * (max_len - len(f["attention_mask"][:max_len]))
                for f in features
            ])

        # Multi-hot labels
        label_ids = torch.zeros((len(features), num_labels))
        for i, f in enumerate(features):
            for label_id in f["label_ids"]:
                label_ids[i, label_id] = 1
        batch["labels"] = label_ids

        return batch

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=4,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare with accelerator
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Training
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            epoch_loss += loss.item()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                avg_loss = epoch_loss / (step + 1)
                progress_bar.set_postfix(loss=avg_loss)

                # Log to wandb
                if args.use_wandb and accelerator.is_main_process and completed_steps % 50 == 0:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/step": completed_steps,
                        "train/epoch": epoch + step / len(train_dataloader),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                    })

            if completed_steps >= max_train_steps:
                break

        # Evaluation
        model.eval()
        all_preds_raw = []
        all_labels = []

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                outputs = model(**batch)

            preds_raw = outputs.logits.sigmoid().cpu()
            all_preds_raw.extend(list(preds_raw))
            all_labels.extend(list(batch["labels"].cpu().numpy()))

        all_preds_raw = np.stack(all_preds_raw)
        all_labels = np.stack(all_labels)

        # Search for optimal threshold instead of hardcoding 0.5
        best_threshold = 0.1
        best_threshold_f1 = 0.0
        labels_flat = all_labels.ravel()
        for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            preds_t = (all_preds_raw > t).astype(np.int8).ravel()
            f1_t = micro_f1(preds_t, labels_flat)
            if np.isnan(f1_t):
                f1_t = 0.0
            if f1_t > best_threshold_f1:
                best_threshold_f1 = f1_t
                best_threshold = t
        logger.info(f"  Best threshold: {best_threshold} (f1_micro={best_threshold_f1:.4f})")

        all_preds = (all_preds_raw > best_threshold).astype(int)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)

        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        logger.info(f"  Loss: {epoch_loss / len(train_dataloader):.4f}")
        logger.info(f"  Metrics: {metrics}")

        # Log eval metrics to wandb
        if args.use_wandb and accelerator.is_main_process:
            wandb.log({
                "eval/loss": epoch_loss / len(train_dataloader),
                "eval/f1_micro": metrics.get('f1_micro', 0),
                "eval/f1_macro": metrics.get('f1_macro', 0),
                "eval/precision_micro": metrics.get('prec_micro', 0),
                "eval/recall_micro": metrics.get('rec_micro', 0),
                "eval/auc_micro": metrics.get('auc_micro', 0),
                "eval/epoch": epoch + 1,
            })

        # Save checkpoint every epoch (overwrites previous)
        current_f1 = metrics.get('f1_micro', 0)
        if np.isnan(current_f1):
            current_f1 = 0.0
        if args.save_every_epoch:
            logger.info(f"  Saving epoch {epoch + 1} checkpoint...")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(args.output_dir)
            # Save epoch info
            if accelerator.is_main_process:
                with open(os.path.join(args.output_dir, "training_state.txt"), "w") as f:
                    f.write(f"epoch: {epoch + 1}\n")
                    f.write(f"f1_micro: {current_f1:.4f}\n")
                    f.write(f"best_f1: {best_f1:.4f}\n")

        # Save best model to separate directory
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            logger.info(f"  New best F1: {best_f1:.4f}, saving to {best_dir}...")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(best_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(best_dir)
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")

        # Early stopping check
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {patience_counter} epochs)")
            break

    # Final evaluation with different thresholds
    logger.info("\n***** Final Evaluation with Different Thresholds *****")
    for t in [0.2, 0.3, 0.4, 0.5]:
        all_preds_t = (all_preds_raw > t).astype(int)
        metrics_t = all_metrics(yhat=all_preds_t, y=all_labels, yhat_raw=all_preds_raw)
        logger.info(f"Threshold {t}: {metrics_t}")

    logger.info(f"\nTraining completed. Best F1: {best_f1:.4f}")
    logger.info(f"Model saved to: {args.output_dir}")

    # Finish wandb run
    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
