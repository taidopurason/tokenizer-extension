import logging
import os
import sys

from transformers import AutoTokenizer

from tokenizer_extension.extension import extend_tokenizer
from tokenizer_extension.utils import read_json, get_vocab_and_merges


def read_sentencepiece_vocab(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.split()[0] for line in f]


def extend(
        tokenizer_path: str,
        output_path: str,
        extension_path: str,
        n_tokens: int,
        extension_method: str,
):
    if extension_method == "sentencepiece":
        sp_vocab = read_sentencepiece_vocab(extension_path)
        new_vocab = {piece: idx for idx, piece in enumerate(sp_vocab)}
    elif extension_method == "continued-training":
        new_vocab = read_json(f"{extension_path}/vocab.json")
    elif extension_method == "hf-training":
        new_vocab = get_vocab_and_merges(
            AutoTokenizer.from_pretrained(extension_path)
        )[0]
    else:
        raise ValueError(f"Unknown extension method: {extension_method}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer = extend_tokenizer(tokenizer, new_vocab, n_tokens=n_tokens)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(extend)
