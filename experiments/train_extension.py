import logging
import os
import sys
import gc
from typing import Optional

import unicodedata
from datasets import load_from_disk
from transformers import AutoTokenizer

from tokenizer_extension.extension import extend_tokenizer
from tokenizer_extension.sentencepiece_utils import train_coverage_extension
from tokenizer_extension.train_vocab_extension import train_vocab_extension
from tokenizer_extension.utils import write_json, get_vocab_and_merges, get_ordered_vocab


def extend_model(
        input_path: str,
        output_path: str,
        tokenizer_path: str,
        vocab_size: int = 64000,
        is_sentencepiece: bool = False,
        required_sp_coverage: Optional[float] = None,
        nfkc_normalize: bool = False,
        sp_legacy_implementation: bool = True,
        max_token_length: Optional[int] = None,
):
    train_docs = load_from_disk(input_path)["text"]
    if nfkc_normalize:
        train_docs = [unicodedata.normalize('NFKC', x) for x in train_docs]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    ext_coverage_vocab = None
    if is_sentencepiece and required_sp_coverage is not None:
        logging.info("Training coverage extension")
        vocab, _ = get_vocab_and_merges(tokenizer)
        ext_coverage_vocab = train_coverage_extension(
            data=train_docs,
            vocab=vocab,
            coverage=required_sp_coverage,
        )
        if ext_coverage_vocab is not None and len(ext_coverage_vocab) > 0:
            logging.info(f"Adding {len(ext_coverage_vocab)} tokens to the tokenizer")
            tokenizer = extend_tokenizer(tokenizer, ext_coverage_vocab, alphabet=[])
        else:
            ext_coverage_vocab = None

    gc.collect()
    extension_tokens = train_vocab_extension(
        tokenizer=tokenizer,
        corpus=train_docs,
        extension_size=vocab_size,
        is_sentencepiece=is_sentencepiece,
        max_token_length=max_token_length,
        sp_legacy_implementation=sp_legacy_implementation,
    )
    if ext_coverage_vocab is not None:
        cov_vocab = get_ordered_vocab(ext_coverage_vocab)
        ext_vocab = get_ordered_vocab(extension_tokens["vocab"])
        extension_tokens["vocab"] = {x: idx for idx, x in enumerate(cov_vocab + ext_vocab)}

    logging.info(f"Saving added vocabulary to {output_path}")
    write_json(extension_tokens["vocab"], os.path.join(output_path, "vocab.json"))
    write_json([" ".join(x) for x in extension_tokens["merges"]], os.path.join(output_path, "merges.json"))


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(extend_model)
