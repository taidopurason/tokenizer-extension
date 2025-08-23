import logging
import os
import sys
import time

from transformers import AutoTokenizer

from tokenizer_extension.sentencepiece_utils import train_sentencepiece_from_model
from tokenizer_extension.utils import batch_iterator, update_postprocessor_special_tokens
from datasets import load_from_disk


def hf_train_from_scratch(
        input_path: str,
        output_path: str,
        tokenizer_path: str,
        vocab_size: int = 64000,
):
    train_docs = load_from_disk(input_path)["text"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logging.info(
        f"Training a new tokenizer from {tokenizer_path} with vocab size {vocab_size} and saving to {output_path}"
    )
    start = time.time()
    train_batch_iterator = batch_iterator(train_docs)
    new_tokenizer = update_postprocessor_special_tokens(
        tokenizer.train_new_from_iterator(train_batch_iterator, vocab_size)
    )
    end = time.time()
    logging.info(f"Training took {end - start} seconds")
    new_tokenizer.save_pretrained(output_path)


def train_from_scratch(
        input_path: str,
        output_path: str,
        tokenizer_path: str,
        vocab_size: int = 64000,
        implementation: str = "huggingface",
        sp_num_threads: int = 16,
):
    if implementation == "huggingface":
        hf_train_from_scratch(
            input_path=input_path,
            output_path=output_path,
            tokenizer_path=tokenizer_path,
            vocab_size=vocab_size,
        )
    elif implementation == "sentencepiece":
        train_sentencepiece_from_model(
            input_path=input_path,
            model_prefix=output_path,
            tokenizer_path=tokenizer_path,
            vocab_size=vocab_size,
            num_threads=sp_num_threads
        )
    else:
        raise ValueError(f"Unknown implementation: {implementation}")


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(train_from_scratch)
