import logging
import os
import sys

from transformers import AutoTokenizer

from tokenizer_extension.extension import extend_tokenizer, extend_sp_model
from tokenizer_extension.utils import read_json, get_vocab_and_merges
from tokenizer_extension.sentencepiece_utils import reorder_sp_vocab, read_sentencepiece_vocab


def extend(
        tokenizer_path: str,
        output_path: str,
        extension_path: str,
        n_tokens: int,
        extension_method: str = "continued-training",
        tokenizer_implementation: str = "huggingface",
        sp_add_chars_first: bool = False,
        is_sentencepiece: bool = False,
        keep_added_token_positions: bool = False,
):
    new_merges = None
    if extension_method == "continued-training":
        new_vocab = read_json(f"{extension_path}/vocab.json")
        new_merges = [tuple(x.split(" ")) for x in read_json(f"{extension_path}/merges.json")]
    elif extension_method == "hf-model":
        new_vocab = get_vocab_and_merges(
            AutoTokenizer.from_pretrained(extension_path)
        )[0]
    elif extension_method == "sp-model":
        is_sentencepiece = True
        sp_vocab = read_sentencepiece_vocab(extension_path)
        new_vocab = {piece: idx for idx, piece in enumerate(sp_vocab)}
        if sp_add_chars_first:
            new_vocab = reorder_sp_vocab(new_vocab)
    else:
        raise ValueError(f"Unknown extension method: {extension_method}")

    logging.info(f"Extending tokenizer from {tokenizer_path} and saving to {output_path}")

    if tokenizer_implementation == "sentencepiece":
        is_sentencepiece = True
        extend_sp_model(
            tokenizer_prefix=tokenizer_path,
            extension_vocab=new_vocab,
            out_prefix=output_path,
            n_tokens=n_tokens,
        )
    elif tokenizer_implementation == "huggingface":
        alphabet = [] if is_sentencepiece else "byte"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer = extend_tokenizer(
            tokenizer,
            new_vocab=new_vocab,
            new_merges=new_merges,
            n_tokens=n_tokens,
            alphabet=alphabet,
            keep_added_token_positions=keep_added_token_positions,
        )
        tokenizer.save_pretrained(output_path)
    else:
        raise ValueError(f"Unknown tokenizer implementation: {tokenizer_implementation}")


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(extend)
