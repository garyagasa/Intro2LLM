from __future__ import annotations

from pathlib import Path

from transformers import BertForMaskedLM, BertTokenizerFast, pipeline

DEFAULT_EXAMPLES = [
    "Today's most trending hashtags on [MASK] is Donald Trump",
    "The [MASK] was cloudy yesterday, but today it's rainy.",
]


def run_fill_mask_examples(
    model_dir: Path,
    tokenizer_dir: Path,
    examples: list[str] | None = None,
    top_k: int = 5,
) -> None:
    """Run a small fill-mask demo after training completes."""

    model = BertForMaskedLM.from_pretrained(str(model_dir))
    tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_dir))
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    for example in examples or DEFAULT_EXAMPLES:
        print(example)
        for prediction in fill_mask(example, top_k=top_k):
            print(f"{prediction['sequence']}, confidence: {prediction['score']}")
        print("=" * 50)
