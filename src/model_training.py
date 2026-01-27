"""
Text-to-SQL Model Training Pipeline

Production-grade training infrastructure for T5/BART fine-tuning
with schema-aware SQL generation.
"""

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

# Optional integrations
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture and loading."""
    model_name: str = "t5-base"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False

    # Architecture choices
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    label_smoothing: float = 0.1

    # Generation config
    num_beams: int = 5
    max_length: int = 256
    min_length: int = 1
    no_repeat_ngram_size: int = 0
    length_penalty: float = 1.0
    early_stopping: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    output_dir: str = "./models/t5_finetuned"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    warmup_ratio: float = 0.0  # Alternative to warmup_steps

    # Optimization
    max_grad_norm: float = 1.0
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Precision
    fp16: bool = True
    bf16: bool = False  # Use for A100/H100 GPUs
    fp16_full_eval: bool = False

    # Checkpointing
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Evaluation
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    logging_steps: int = 100
    logging_first_step: bool = True

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # Memory optimization
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"  # or "adamw_hf", "adafactor"

    # Distributed training
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Reproducibility
    seed: int = 42

    # Logging
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None


class Text2SQLTrainer:
    """
    Production training pipeline for Text-to-SQL models.

    Supports:
    - T5, BART, and compatible seq2seq models
    - Mixed precision training (fp16/bf16)
    - Gradient checkpointing for memory efficiency
    - Distributed training (DDP)
    - WandB and TensorBoard logging
    - Early stopping with customizable patience
    - Checkpoint management and resumption
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig
    ):
        self.model_config = model_config
        self.training_config = training_config

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[Seq2SeqTrainer] = None

        # Set seed for reproducibility
        self._set_seed(training_config.seed)

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def setup_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Initialize model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.model_config.model_name}")

        tokenizer_name = self.model_config.tokenizer_name or self.model_config.model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=self.model_config.cache_dir,
            use_fast=self.model_config.use_fast_tokenizer,
            trust_remote_code=self.model_config.trust_remote_code
        )

        # Load model configuration
        config = AutoConfig.from_pretrained(
            self.model_config.model_name,
            cache_dir=self.model_config.cache_dir,
            trust_remote_code=self.model_config.trust_remote_code
        )

        # Update config with our settings
        if hasattr(config, "dropout_rate"):
            config.dropout_rate = self.model_config.dropout_rate
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = self.model_config.attention_dropout

        # Load model
        if "t5" in self.model_config.model_name.lower():
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_config.model_name,
                config=config,
                cache_dir=self.model_config.cache_dir,
                trust_remote_code=self.model_config.trust_remote_code
            )
        elif "bart" in self.model_config.model_name.lower():
            self.model = BartForConditionalGeneration.from_pretrained(
                self.model_config.model_name,
                config=config,
                cache_dir=self.model_config.cache_dir,
                trust_remote_code=self.model_config.trust_remote_code
            )
        else:
            # Generic seq2seq model loading
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_config.model_name,
                config=config,
                cache_dir=self.model_config.cache_dir,
                trust_remote_code=self.model_config.trust_remote_code
            )

        # Enable gradient checkpointing if configured
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return self.model, self.tokenizer

    def get_data_collator(self) -> DataCollatorForSeq2Seq:
        """
        Create data collator for dynamic padding.

        Returns:
            DataCollatorForSeq2Seq instance
        """
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,  # Ignore in loss computation
            pad_to_multiple_of=8 if self.training_config.fp16 else None
        )

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        """
        Build Seq2Seq training arguments.

        Returns:
            Configured Seq2SeqTrainingArguments
        """
        cfg = self.training_config

        # Determine device capabilities
        use_fp16 = cfg.fp16 and torch.cuda.is_available()
        use_bf16 = cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        if use_bf16:
            use_fp16 = False  # Prefer bf16 over fp16

        return Seq2SeqTrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_steps=cfg.warmup_steps if cfg.warmup_steps > 0 else None,
            warmup_ratio=cfg.warmup_ratio if cfg.warmup_steps == 0 else 0.0,
            max_grad_norm=cfg.max_grad_norm,
            adam_epsilon=cfg.adam_epsilon,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,

            # Precision
            fp16=use_fp16,
            bf16=use_bf16,
            fp16_full_eval=cfg.fp16_full_eval,

            # Checkpointing
            save_strategy=cfg.save_strategy,
            save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model=cfg.metric_for_best_model,
            greater_is_better=cfg.greater_is_better,

            # Evaluation
            eval_strategy=cfg.eval_strategy,
            eval_steps=cfg.eval_steps if cfg.eval_strategy == "steps" else None,
            logging_steps=cfg.logging_steps,
            logging_first_step=cfg.logging_first_step,

            # Generation settings for eval
            predict_with_generate=True,
            generation_max_length=self.model_config.max_length,
            generation_num_beams=self.model_config.num_beams,

            # Optimization
            optim=cfg.optim,
            gradient_checkpointing=cfg.gradient_checkpointing,

            # Distributed
            ddp_find_unused_parameters=cfg.ddp_find_unused_parameters,
            dataloader_num_workers=cfg.dataloader_num_workers,
            dataloader_pin_memory=cfg.dataloader_pin_memory,

            # Reproducibility
            seed=cfg.seed,
            data_seed=cfg.seed,

            # Logging
            report_to=cfg.report_to,
            run_name=cfg.run_name,

            # Misc
            remove_unused_columns=True,
            label_smoothing_factor=self.model_config.label_smoothing,
            push_to_hub=False,
        )

    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            eval_preds: EvalPrediction object with predictions and labels

        Returns:
            Dictionary of metric names to values
        """
        predictions, labels = eval_preds

        # Decode predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Replace -100 with pad token id for decoding
        predictions = np.where(
            predictions != -100,
            predictions,
            self.tokenizer.pad_token_id
        )
        labels = np.where(
            labels != -100,
            labels,
            self.tokenizer.pad_token_id
        )

        decoded_preds = self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Normalize for comparison
        decoded_preds = [pred.strip().lower() for pred in decoded_preds]
        decoded_labels = [label.strip().lower() for label in decoded_labels]

        # Exact match accuracy
        exact_matches = sum(
            1 for pred, label in zip(decoded_preds, decoded_labels)
            if pred == label
        )
        exact_match_accuracy = exact_matches / len(decoded_preds) if decoded_preds else 0.0

        # Token-level accuracy (for partial credit)
        def token_accuracy(pred: str, label: str) -> float:
            pred_tokens = set(pred.split())
            label_tokens = set(label.split())
            if not label_tokens:
                return 1.0 if not pred_tokens else 0.0
            intersection = pred_tokens & label_tokens
            return len(intersection) / len(label_tokens)

        token_accuracies = [
            token_accuracy(pred, label)
            for pred, label in zip(decoded_preds, decoded_labels)
        ]
        avg_token_accuracy = np.mean(token_accuracies) if token_accuracies else 0.0

        return {
            "exact_match": exact_match_accuracy,
            "token_accuracy": avg_token_accuracy,
        }

    def setup_trainer(
        self,
        train_dataset,
        eval_dataset,
        compute_metrics_fn=None
    ) -> Seq2SeqTrainer:
        """
        Initialize the Seq2Seq trainer.

        Args:
            train_dataset: Tokenized training dataset
            eval_dataset: Tokenized evaluation dataset
            compute_metrics_fn: Optional custom metrics function

        Returns:
            Configured Seq2SeqTrainer
        """
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()

        training_args = self.get_training_arguments()
        data_collator = self.get_data_collator()

        # Callbacks
        callbacks = []
        if self.training_config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold
                )
            )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn or self.compute_metrics,
            callbacks=callbacks
        )

        return self.trainer

    def train(
        self,
        train_dataset,
        eval_dataset,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute training loop.

        Args:
            train_dataset: Tokenized training dataset
            eval_dataset: Tokenized evaluation dataset
            resume_from_checkpoint: Path to checkpoint for resumption

        Returns:
            Training metrics dictionary
        """
        if self.trainer is None:
            self.setup_trainer(train_dataset, eval_dataset)

        logger.info("Starting training...")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num epochs = {self.training_config.num_train_epochs}")
        logger.info(f"  Batch size = {self.training_config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation = {self.training_config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {len(train_dataset) // (self.training_config.per_device_train_batch_size * self.training_config.gradient_accumulation_steps) * self.training_config.num_train_epochs}")

        # Train
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config.output_dir)

        # Save training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        logger.info(f"Training complete. Model saved to {self.training_config.output_dir}")

        return metrics

    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """
        Run evaluation on dataset.

        Args:
            eval_dataset: Tokenized evaluation dataset

        Returns:
            Evaluation metrics dictionary
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")

        logger.info("Running evaluation...")
        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        return metrics

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save model and tokenizer to directory.

        Args:
            output_dir: Output directory (uses config default if None)
        """
        save_dir = output_dir or self.training_config.output_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save configs
        config_path = Path(save_dir) / "training_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "model_config": vars(self.model_config),
                "training_config": {
                    k: v for k, v in vars(self.training_config).items()
                    if not callable(v)
                }
            }, f, indent=2, default=str)

        logger.info(f"Model saved to {save_dir}")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto"
    ) -> "Text2SQLTrainer":
        """
        Load trainer from saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to load model on

        Returns:
            Text2SQLTrainer instance with loaded model
        """
        checkpoint_path = Path(checkpoint_path)

        # Load configs if available
        config_path = checkpoint_path / "training_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                configs = json.load(f)
            model_config = ModelConfig(**configs.get("model_config", {}))
            training_config = TrainingConfig(**configs.get("training_config", {}))
        else:
            model_config = ModelConfig()
            training_config = TrainingConfig()

        # Update model name to checkpoint path
        model_config.model_name = str(checkpoint_path)

        trainer_instance = cls(model_config, training_config)
        trainer_instance.setup_model_and_tokenizer()

        # Move to device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer_instance.model.to(device)

        return trainer_instance


class CustomTrainingLoop:
    """
    Custom training loop for advanced use cases.

    Use this when you need fine-grained control over the training process
    that the Trainer API doesn't provide.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None

    def _setup_optimizer(self):
        """Setup AdamW optimizer with weight decay."""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            betas=(self.config.adam_beta1, self.config.adam_beta2)
        )

    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler with warmup."""
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        scheduler,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            scheduler: Learning rate scheduler
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not self._is_main_process()
        )

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

            total_loss += loss.item() * self.config.gradient_accumulation_steps

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                scheduler.step()
                self.optimizer.zero_grad()

            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"eval_loss": avg_loss}

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Validation data loader
            num_epochs: Number of epochs (uses config default if None)

        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.num_train_epochs
        num_training_steps = len(train_dataloader) * num_epochs

        scheduler = self._setup_scheduler(num_training_steps)

        history = {"train_loss": [], "eval_loss": []}
        best_eval_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_dataloader, scheduler, epoch)
            history["train_loss"].append(train_metrics["loss"])

            # Evaluate
            eval_metrics = self.evaluate(eval_dataloader)
            history["eval_loss"].append(eval_metrics["eval_loss"])

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Eval Loss: {eval_metrics['eval_loss']:.4f}"
            )

            # Checkpointing
            if eval_metrics["eval_loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval_loss"]
                patience_counter = 0
                self._save_checkpoint(epoch, best_eval_loss)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return history

    def _save_checkpoint(self, epoch: int, eval_loss: float):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir / "best_model")
        self.tokenizer.save_pretrained(output_dir / "best_model")

        # Save checkpoint info
        checkpoint_info = {
            "epoch": epoch,
            "eval_loss": eval_loss
        }
        with open(output_dir / "checkpoint_info.json", "w") as f:
            json.dump(checkpoint_info, f)

    def _is_main_process(self) -> bool:
        """Check if this is the main process (for distributed training)."""
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory information for monitoring."""
    if not torch.cuda.is_available():
        return {"available": False}

    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": []
    }

    for i in range(torch.cuda.device_count()):
        device_info = {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
            "allocated_memory_gb": torch.cuda.memory_allocated(i) / 1e9,
            "cached_memory_gb": torch.cuda.memory_reserved(i) / 1e9,
        }
        info["devices"].append(device_info)

    return info


if __name__ == "__main__":
    # Quick test of configuration
    model_config = ModelConfig(model_name="t5-base")
    training_config = TrainingConfig(
        output_dir="./models/test",
        num_train_epochs=1,
        per_device_train_batch_size=2
    )

    print("Model Config:", model_config)
    print("Training Config:", training_config)
    print("GPU Info:", get_gpu_memory_info())
