import logging
from typing import Optional

from tokenizer_extension.extension import extend_tokenizer
from tokenizer_extension.utils import get_vocab_and_merges
from datasets import load_from_disk
from collections import Counter
from tqdm import tqdm
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

    counter = Counter()
    for doc in tqdm(data):
        counter.update(doc)

    if implementation == "hf":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab, _ = get_vocab_and_merges(tokenizer)
    elif implementation == "sentencepiece":
        from tokenizer_extension.sentencepiece_utils import read_sentencepiece_vocab
        vocab = read_sentencepiece_vocab(tokenizer_path + ".vocab")
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    total_characters = sum(counter.values())
    in_vocab_characters = sum(v for x, v in counter.keys() if x in vocab)
    initial_coverage = in_vocab_characters / total_characters
    logging.info(f"The vocabulary coverage is {initial_coverage}")

    if initial_coverage >= coverage:
        logging.info(f"Coverage already achieved, no need to extend")
        return

    sorted_characters = sorted(
        [(k, c) for k, c in counter.items() if k not in vocab],
        key=lambda x: x[1],
        reverse=True
    )
    n_tokens = 0
    for _, c in sorted_characters:
        in_vocab_characters += c
        n_tokens += 1
        if in_vocab_characters / total_characters >= coverage:
            break

    logging.info(f"Number of tokens to add: {n_tokens}")

    if max_tokens is not None and n_tokens > max_tokens:
        logging.info(f"Number of tokens to add exceeds max_tokens, capping at {max_tokens}")
        n_tokens = max_tokens

    sorted_characters = sorted_characters[:n_tokens]
    for x in sorted_characters:
        logging.info(f"Adding {x[0]} with frequency {x[1]}")

    new_vocab = {x[0]: idx for idx, x in enumerate(sorted_characters)}
    if implementation == "hf":
        new_tokenizer = extend_tokenizer(AutoTokenizer.from_pretrained(tokenizer_path), new_vocab, n_tokens=n_tokens)
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
