import logging
import os
import sys
import time
from typing import Optional

from transformers import AutoTokenizer

from tokenizer_extension.pruning import PRUNER_REGISTRY, TrainablePrunerBase
from datasets import load_from_disk


def train_pruner(
        output_path: str,
        tokenizer_path: str,
        pruner_name: str,
        input_path: Optional[str] = None,
):
    if input_path is not None:
        train_docs = load_from_disk(input_path)["text"]
    else:
        train_docs = None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logging.info(
        f"Calculating pruning orders {tokenizer_path} and saving to {output_path}"
    )
    start = time.time()
    if pruner_name not in PRUNER_REGISTRY:
        raise ValueError(f"Unknown pruner: {pruner_name}. Available pruners: {list(PRUNER_REGISTRY.keys())}")
    pruner = PRUNER_REGISTRY[pruner_name]()
    if isinstance(pruner, TrainablePrunerBase) and train_docs is None:
        raise ValueError(f"Pruner {pruner_name} requires training data, but no input_path was provided")

    pruner.train(tokenizer, train_docs).save(output_path)

    end = time.time()
    logging.info(f"Calculation took {end - start} seconds")


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(train_pruner)
