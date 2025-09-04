import logging
import os
import sys
import time
from typing import Optional

from transformers import AutoTokenizer

from tokenizer_extension.pruning import calculate_orders
from tokenizer_extension.utils import write_json
from datasets import load_from_disk


def train_pruner(
        input_path: str,
        output_path: str,
        tokenizer_path: str,
        calculate_token_frequency: bool = True,
        calculate_merge_based_pruning: bool = True,
        return_counts: bool = True,
        ignore_merges: Optional[bool] = None,
):
    train_docs = load_from_disk(input_path)["text"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logging.info(
        f"Calculating pruning orders {tokenizer_path} and saving to {output_path}"
    )
    start = time.time()

    pruning_orders = calculate_orders(
        texts=train_docs,
        tokenizer=tokenizer,
        calculate_token_frequency=calculate_token_frequency,
        calculate_merge_based_pruning=calculate_merge_based_pruning,
        ignore_merges=ignore_merges,
        return_counts=return_counts
    )

    end = time.time()
    logging.info(f"Calculation took {end - start} seconds")
    if "_merge_counts" in pruning_orders:
        pruning_orders["_merge_counts"] = {" ".join(k): v for k, v in pruning_orders["_merge_counts"].items()}
    write_json(pruning_orders, os.path.join(output_path, "pruning_order.json"))


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(train_pruner)
