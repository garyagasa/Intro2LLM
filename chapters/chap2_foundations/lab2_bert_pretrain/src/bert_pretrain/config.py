from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BertPretrainConfig:
    """Central configuration for the full pretraining pipeline."""

    project_root: Path = Path(__file__).resolve().parents[2]
    hf_endpoint: str = "https://hf-mirror.com"

    bookcorpus_name: str = "bookcorpus"
    bookcorpus_config: str = "plain_text"
    wikipedia_name: str = "wikipedia"
    wikipedia_config: str = "20220301.en"
    test_size: float = 0.1
    seed: int = 42

    data_dir: Path = project_root / "data"
    train_text_path: Path = data_dir / "train.txt"
    test_text_path: Path = data_dir / "test.txt"

    artifacts_dir: Path = project_root / "artifacts"
    tokenizer_dir: Path = artifacts_dir / "tokenizer"
    model_dir: Path = artifacts_dir / "model"
    checkpoint_dir: Path = model_dir / "checkpoints"

    vocab_size: int = 30_522
    max_length: int = 512
    truncate_longer_samples: bool = False
    mlm_probability: float = 0.2
    num_proc: int = 8

    num_train_epochs: float = 10.0
    per_device_train_batch_size: int = 10
    gradient_accumulation_steps: int = 8
    per_device_eval_batch_size: int = 64
    logging_steps: int = 1_000
    save_steps: int = 1_000
    eval_steps: int = 1_000
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    save_total_limit: int = 3
    use_bf16: bool = True
    use_tf32: bool = True
    use_wandb: bool = True
    wandb_project: str = "bert-pretrain"
    wandb_run_name: str = "bert-pretrain-h100-cu128"

    @property
    def tokenizer_config_path(self) -> Path:
        return self.tokenizer_dir / "tokenizer_config.json"
