import logging
import os
import sys

from transformers import AutoTokenizer
from transformers.utils.sentencepiece_model_pb2 import ModelProto

from tokenizer_extension.utils import read_json, get_vocab_and_merges, get_ordered_vocab

ILLEGAL_CHARS = {" ", "\n", "\r", ""}


def read_sentencepiece_vocab(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.split()[0] for line in f]


def read_model(path: str):
    model = ModelProto()
    with open(path, "rb") as f:
        model.ParseFromString(f.read())
    return model


def read_tokens(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def add_vocab(tokenizer_prefix, extend_vocab, out_prefix, n_tokens=None):
    model = read_model(f"{tokenizer_prefix}.model")

    score = min(p.score for p in model.pieces) - 1
    vocab = {p.piece for p in model.pieces}

    tokens_to_add = get_ordered_vocab(extend_vocab)
    logging.info(f"Read {len(tokens_to_add)} tokens to add.")
    tokens_to_add = [piece for piece in tokens_to_add if piece not in ILLEGAL_CHARS and piece not in vocab]
    logging.info(f"Removed existing and illegal tokens with remaining {len(tokens_to_add)} tokens to add.")
    if n_tokens is not None:
        tokens_to_add = tokens_to_add[:n_tokens]
    logging.info(f"Adding {len(tokens_to_add)} tokens.")

    for piece in tokens_to_add:
        model.pieces.append(ModelProto.SentencePiece(piece=piece, score=score))
        vocab.add(piece)
        score -= 1

    with open(f"{out_prefix}.model", "wb") as f:
        f.write(model.SerializeToString())

    with open(f"{out_prefix}.vocab", "w", encoding="utf-8") as f:
        for p in model.pieces:
            f.write(f"{p.piece}\t{int(p.score)}\n")


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

    logging.info(f"Loaded tokens to add from {extension_path} ({extension_method}).")
    add_vocab(
        tokenizer_prefix=tokenizer_path,
        extend_vocab=new_vocab,
        out_prefix=output_path,
        n_tokens=n_tokens
    )


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(extend)
