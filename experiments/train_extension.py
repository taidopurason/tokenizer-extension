import os

from datasets import load_from_disk
from transformers import AutoTokenizer

from tokenizer_extension.train_vocab_extension import train_vocab_extension
from tokenizer_extension.utils import write_json


def extend_model(
        input_path: str,
        output_path: str,
        tokenizer_path: str,
        vocab_size: int = 64000,
    is_sentencepiece: bool = False,
):
    train_docs = load_from_disk(input_path)["text"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    extension_tokens = train_vocab_extension(
        tokenizer=tokenizer, corpus=train_docs, extension_size=vocab_size, is_sentencepiece=is_sentencepiece
    )
    write_json(extension_tokens["vocab"], os.path.join(output_path, "vocab.json"))
    write_json([" ".join(x) for x in extension_tokens["merges"]], os.path.join(output_path, "merges.json"))


if __name__ == "__main__":
    import fire
    fire.Fire(extend_model)
