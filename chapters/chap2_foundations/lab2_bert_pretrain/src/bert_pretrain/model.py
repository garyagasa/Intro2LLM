from __future__ import annotations

from transformers import BertConfig, BertForMaskedLM

from .config import BertPretrainConfig


def build_model(config: BertPretrainConfig) -> BertForMaskedLM:
    """Create a BERT masked language model from scratch."""

    model_config = BertConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_length,
    )
    return BertForMaskedLM(config=model_config)
