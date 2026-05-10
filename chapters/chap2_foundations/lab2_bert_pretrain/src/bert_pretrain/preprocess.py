from __future__ import annotations

from functools import partial
from itertools import chain

from datasets import DatasetDict
from transformers import BertTokenizerFast

from .config import BertPretrainConfig


def encode_examples(
    examples,
    tokenizer: BertTokenizerFast,
    max_length: int,
    truncate_longer_samples: bool,
):
    if truncate_longer_samples:
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    return tokenizer(examples["text"], return_special_tokens_mask=True)


def group_texts(examples, max_length: int):
    concatenated_examples = {key: list(chain.from_iterable(examples[key])) for key in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length

    return {
        key: [tokens[index : index + max_length] for index in range(0, total_length, max_length)]
        for key, tokens in concatenated_examples.items()
    }


def tokenize_splits(
    splits: DatasetDict,
    tokenizer: BertTokenizerFast,
    config: BertPretrainConfig,
):
    """Tokenize the train and test splits and build language-model training chunks."""

    encode = partial(
        encode_examples,
        tokenizer=tokenizer,
        max_length=config.max_length,
        truncate_longer_samples=config.truncate_longer_samples,
    )
    remove_columns = splits["train"].column_names

    train_dataset = splits["train"].map(
        encode,
        batched=True,
        remove_columns=remove_columns,
        num_proc=config.num_proc,
        desc="Tokenizing train split",
    )
    eval_dataset = splits["test"].map(
        encode,
        batched=True,
        remove_columns=remove_columns,
        num_proc=config.num_proc,
        desc="Tokenizing eval split",
    )

    if config.truncate_longer_samples:
        return train_dataset, eval_dataset

    train_dataset = train_dataset.map(
        partial(group_texts, max_length=config.max_length),
        batched=True,
        num_proc=config.num_proc,
        desc=f"Grouping train texts in chunks of {config.max_length}",
    )
    eval_dataset = eval_dataset.map(
        partial(group_texts, max_length=config.max_length),
        batched=True,
        num_proc=config.num_proc,
        desc=f"Grouping eval texts in chunks of {config.max_length}",
    )

    return train_dataset, eval_dataset
