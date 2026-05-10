from __future__ import annotations

from pathlib import Path

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast

from .config import BertPretrainConfig

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]


def train_tokenizer(config: BertPretrainConfig, train_files: list[Path]) -> BertTokenizerFast:
    """Train a WordPiece tokenizer and export it in Hugging Face format."""

    config.tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertWordPieceTokenizer(lowercase=True)
    tokenizer.train(
        files=[str(path) for path in train_files],
        vocab_size=config.vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.enable_truncation(max_length=config.max_length)
    tokenizer.save_model(str(config.tokenizer_dir))

    hf_tokenizer = BertTokenizerFast.from_pretrained(str(config.tokenizer_dir), do_lower_case=True)
    hf_tokenizer.save_pretrained(str(config.tokenizer_dir))
    return hf_tokenizer
