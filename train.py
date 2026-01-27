#!/usr/bin/env python3
"""
Text-to-SQL Training CLI

Production training script for fine-tuning T5/BART on Spider dataset.

Usage:
    python train.py --model_name t5-base --epochs 10 --batch_size 8
    python train.py --model_name t5-large --gradient_checkpointing --fp16
    python train.py --model_name facebook/bart-base --dataset wikisql
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import AutoTokenizer

from src.data_preparation import SpiderDataProcessor, WikiSQLDataProcessor
from src.model_training import (
    ModelConfig,
    TrainingConfig,
    Text2SQLTrainer,
    get_gpu_memory_info,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune T5/BART for Text-to-SQL generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_name",
        type=str,
        default="t5-base",
        help="Model name or path (t5-base, t5-large, t5-3b, facebook/bart-base, etc.)"
    )
    model_group.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name (defaults to model_name)"
    )
    model_group.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for generation"
    )
    model_group.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="Maximum input sequence length"
    )
    model_group.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="Maximum output sequence length"
    )
    model_group.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor"
    )

    # Data arguments
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--dataset",
        type=str,
        default="spider",
        choices=["spider", "wikisql", "both"],
        help="Dataset to use for training"
    )
    data_group.add_argument(
        "--data_dir",
        type=str,
        default="./data/spider",
        help="Directory containing dataset files"
    )
    data_group.add_argument(
        "--use_huggingface",
        action="store_true",
        help="Load dataset from Hugging Face Hub"
    )
    data_group.add_argument(
        "--schema_format",
        type=str,
        default="verbose",
        choices=["verbose", "compact", "natural"],
        help="Schema serialization format"
    )
    data_group.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="Number of workers for data preprocessing"
    )

    # Training arguments
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--output_dir",
        type=str,
        default="./models/t5_finetuned",
        help="Output directory for model checkpoints"
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size per device"
    )
    train_group.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Evaluation batch size (defaults to batch_size)"
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate"
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    train_group.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    train_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup ratio (alternative to warmup_steps)"
    )
    train_group.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )

    # Optimization arguments
    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision training"
    )
    optim_group.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 mixed precision (for A100/H100 GPUs)"
    )
    optim_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for memory efficiency"
    )
    optim_group.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        choices=["adamw_torch", "adamw_hf", "adafactor", "adamw_apex_fused"],
        help="Optimizer to use"
    )

    # Checkpointing arguments
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "epoch", "steps"],
        help="Checkpoint save strategy"
    )
    ckpt_group.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (if save_strategy=steps)"
    )
    ckpt_group.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep"
    )
    ckpt_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["no", "epoch", "steps"],
        help="Evaluation strategy"
    )
    eval_group.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (if eval_strategy=steps)"
    )
    eval_group.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_loss",
        help="Metric for selecting best model"
    )
    eval_group.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience (0 to disable)"
    )

    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every N steps"
    )
    log_group.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default=["tensorboard"],
        choices=["tensorboard", "wandb", "none"],
        help="Logging integrations"
    )
    log_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this training run"
    )
    log_group.add_argument(
        "--wandb_project",
        type=str,
        default="text-to-sql",
        help="Weights & Biases project name"
    )

    # System arguments
    sys_group = parser.add_argument_group("System")
    sys_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    sys_group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    sys_group.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Log system info
    logger.info("=" * 60)
    logger.info("Text-to-SQL Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        gpu_info = get_gpu_memory_info()
        for device in gpu_info.get("devices", []):
            logger.info(f"GPU {device['index']}: {device['name']} ({device['total_memory_gb']:.1f} GB)")

    # Setup WandB if requested
    if "wandb" in args.report_to:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args)
            )
        except ImportError:
            logger.warning("wandb not installed, disabling wandb logging")
            args.report_to = [r for r in args.report_to if r != "wandb"]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer first (needed for data preparation)
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name,
        use_fast=True
    )

    # Prepare dataset
    logger.info(f"Preparing {args.dataset} dataset...")

    if args.dataset == "spider":
        processor = SpiderDataProcessor(
            data_dir=args.data_dir,
            max_input_length=args.max_source_length,
            max_target_length=args.max_target_length,
            schema_format=args.schema_format
        )
        dataset = processor.prepare_dataset(use_huggingface=args.use_huggingface)

    elif args.dataset == "wikisql":
        processor = WikiSQLDataProcessor(
            max_input_length=args.max_source_length,
            max_target_length=args.max_target_length
        )
        dataset = processor.prepare_dataset()

    elif args.dataset == "both":
        # Combine Spider and WikiSQL
        spider_processor = SpiderDataProcessor(
            data_dir=args.data_dir,
            max_input_length=args.max_source_length,
            max_target_length=args.max_target_length,
            schema_format=args.schema_format
        )
        wikisql_processor = WikiSQLDataProcessor(
            max_input_length=args.max_source_length,
            max_target_length=args.max_target_length
        )

        spider_dataset = spider_processor.prepare_dataset(use_huggingface=args.use_huggingface)
        wikisql_dataset = wikisql_processor.prepare_dataset()

        # Concatenate training sets
        from datasets import concatenate_datasets, DatasetDict
        combined_train = concatenate_datasets([
            spider_dataset["train"],
            wikisql_dataset["train"]
        ])
        dataset = DatasetDict({
            "train": combined_train,
            "validation": spider_dataset["validation"]  # Use Spider validation
        })

    logger.info(f"Train examples: {len(dataset['train'])}")
    logger.info(f"Validation examples: {len(dataset['validation'])}")

    # Tokenize dataset
    logger.info("Tokenizing dataset...")

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=args.max_source_length,
            truncation=True,
            padding=False
        )
        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=args.max_target_length,
            truncation=True,
            padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )

    # Configure model
    model_config = ModelConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        num_beams=args.num_beams,
        max_length=args.max_target_length,
        label_smoothing=args.label_smoothing
    )

    # Configure training
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        metric_for_best_model=args.metric_for_best_model,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to if "none" not in args.report_to else [],
        run_name=args.run_name
    )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Text2SQLTrainer(model_config, training_config)
    trainer.setup_model_and_tokenizer()

    # Setup trainer with datasets
    trainer.setup_trainer(
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    # Train
    logger.info("Starting training...")
    metrics = trainer.train(
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # Final evaluation
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate(tokenized_dataset["validation"])

    # Log final metrics
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Final train loss: {metrics.get('train_loss', 'N/A')}")
    logger.info(f"Final eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
    logger.info(f"Exact match accuracy: {eval_metrics.get('eval_exact_match', 'N/A')}")
    logger.info(f"Model saved to: {args.output_dir}")

    # Save final metrics
    import json
    metrics_path = output_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "train_metrics": metrics,
            "eval_metrics": eval_metrics,
            "args": vars(args)
        }, f, indent=2, default=str)

    logger.info(f"Metrics saved to: {metrics_path}")

    # Cleanup
    if "wandb" in args.report_to:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
