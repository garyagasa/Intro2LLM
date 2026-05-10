from __future__ import annotations

import os

from src.bert_pretrain.config import BertPretrainConfig
from src.bert_pretrain.data import load_corpus, save_text_split
from src.bert_pretrain.inference import run_fill_mask_examples
from src.bert_pretrain.model import build_model
from src.bert_pretrain.preprocess import tokenize_splits
from src.bert_pretrain.tokenizer import train_tokenizer
from src.bert_pretrain.trainer import build_trainer, train_and_save


def run_pretraining(config: BertPretrainConfig | None = None) -> None:
    """Run the full BERT pretraining workflow."""

    config = config or BertPretrainConfig()
    os.environ["HF_ENDPOINT"] = config.hf_endpoint

    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    splits = load_corpus(config)
    save_text_split(splits["train"], config.train_text_path)
    save_text_split(splits["test"], config.test_text_path)

    tokenizer = train_tokenizer(config, [config.train_text_path])
    train_dataset, eval_dataset = tokenize_splits(splits, tokenizer, config)

    model = build_model(config)
    trainer = build_trainer(config, model, tokenizer, train_dataset, eval_dataset)
    train_and_save(trainer, config.model_dir)

    run_fill_mask_examples(config.model_dir, config.tokenizer_dir)


if __name__ == "__main__":
    run_pretraining()
