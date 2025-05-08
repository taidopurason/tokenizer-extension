import logging
from typing import Optional

from tokenizer_extension.extension import extend_tokenizer
from tokenizer_extension.sentencepiece_utils import train_coverage_extension
from tokenizer_extension.utils import get_vocab_and_merges
from datasets import load_from_disk
from transformers import AutoTokenizer


def extend_coverage(
        input_path: str,
        output_path: str,
        tokenizer_path: str,
        coverage: float = 0.9995,
        max_tokens: Optional[int] = None,
        implementation: str = "hf"
):
    data = load_from_disk(input_path)["text"]

    if implementation == "hf":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab, _ = get_vocab_and_merges(tokenizer)
    elif implementation == "sentencepiece":
        from tokenizer_extension.sentencepiece_utils import read_sentencepiece_vocab
        vocab = read_sentencepiece_vocab(tokenizer_path + ".vocab")
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    new_vocab = train_coverage_extension(
        data=data,
        vocab=vocab,
        coverage=coverage,
        max_tokens=max_tokens,
    )
    if implementation == "hf":
        new_tokenizer = extend_tokenizer(AutoTokenizer.from_pretrained(tokenizer_path), new_vocab)
        new_tokenizer.save_pretrained(output_path)
    elif implementation == "sentencepiece":
        from tokenizer_extension.sentencepiece_utils import extend_vocab
        extend_vocab(tokenizer_path, new_vocab, output_path)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    fire.Fire(extend_coverage)
