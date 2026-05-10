from __future__ import annotations

import os
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from .config import BertPretrainConfig


def load_corpus(config: BertPretrainConfig) -> DatasetDict:
    """Load and merge the source corpora, then split them into train and test sets."""

    os.environ["HF_ENDPOINT"] = config.hf_endpoint

    bookcorpus = load_dataset(config.bookcorpus_name, config.bookcorpus_config, split="train")
    wikipedia = load_dataset(config.wikipedia_name, config.wikipedia_config, split="train")
    wikipedia = wikipedia.remove_columns([column for column in wikipedia.column_names if column != "text"])

    merged = concatenate_datasets([bookcorpus, wikipedia])
    return merged.train_test_split(test_size=config.test_size, seed=config.seed)


def save_text_split(dataset: Dataset, output_path: Path) -> None:
    """Persist a text-only dataset split to disk as one document per line."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for text in dataset["text"]:
            handle.write(f"{text}\n")
