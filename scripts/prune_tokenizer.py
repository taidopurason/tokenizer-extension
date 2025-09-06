import logging
import os
import sys

from transformers import AutoTokenizer

from tokenizer_extension.pruning import PretrainedPruner


def prune(
        tokenizer_path: str,
        output_path: str,
        prune_order_path: str,
        n_tokens: int,
):
    pruner = PretrainedPruner.load(prune_order_path)
    logging.info(f"Pruning with {pruner.name} from {prune_order_path} ({n_tokens}): {tokenizer_path}")
    tokenizer = pruner.prune(AutoTokenizer.from_pretrained(tokenizer_path), n_tokens)
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
