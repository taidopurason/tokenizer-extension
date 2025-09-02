import logging
import os
import sys

from transformers import AutoTokenizer

from tokenizer_extension.pruning import prune_tokenizer
from tokenizer_extension.utils import read_json


def prune(
        tokenizer_path: str,
        output_path: str,
        prune_order_path: str,
        prune_order_name: str,
        n_tokens: int,
):
    logging.info(f"Pruning with {prune_order_name} from {prune_order_path} ({n_tokens}): {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab = read_json(os.path.join(prune_order_path, "pruning_order.json"))[prune_order_name]

    tokenizer = prune_tokenizer(
        tokenizer,
        vocab,
        n_tokens,
        ignore_special=True,
        ignore_added=True,
    )
    logging.info(f"Saving pruned tokenizer to {output_path}")
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(prune)
