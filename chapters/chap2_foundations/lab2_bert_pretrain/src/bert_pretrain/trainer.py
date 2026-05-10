from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from .config import BertPretrainConfig


def build_trainer(
    config: BertPretrainConfig,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
) -> Trainer:
    """Construct the Hugging Face Trainer with H100-friendly defaults."""

    use_bf16 = config.use_bf16 and torch.cuda.is_available()
    use_tf32 = config.use_tf32 and torch.cuda.is_available()

    if config.use_wandb:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_NAME"] = config.wandb_run_name

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=str(config.checkpoint_dir),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        save_total_limit=config.save_total_limit,
        bf16=use_bf16,
        tf32=use_tf32,
        dataloader_num_workers=config.num_proc,
        report_to=["wandb"] if config.use_wandb else ["none"],
        run_name=config.wandb_run_name,
        seed=config.seed,
    )

    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def train_and_save(trainer: Trainer, model_dir: Path) -> None:
    """Train the model and persist the final weights to a dedicated model directory."""
    try:
        trainer.train()
    except Exception as e:
        # If a DataLoader worker fails, retry in the main process with single worker
        import traceback

        print("Error during trainer.train():\n", e)
        traceback.print_exc()
        print("Retrying training with dataloader_num_workers=0 to reproduce the error in main process...")
        # mutate training args for single-process dataloader
        try:
            trainer.args.dataloader_num_workers = 0
        except Exception:
            pass
        trainer.train()

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
