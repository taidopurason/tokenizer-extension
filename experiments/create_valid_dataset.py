import logging
import os
import sys

from datasets import load_dataset, Dataset

from tokenizer_extension.utils import budget_iterator


def get_culturax_ds(lang="et", streaming=True):
    extra_shuffle_args = {}
    if streaming:
        extra_shuffle_args = {"buffer_size": 1000}

    return load_dataset(
        "uonlp/CulturaX", lang, split="train", streaming=streaming
    ).shuffle(seed=42, **extra_shuffle_args).take(10000)


def get_fineweb_ds(lang="ekk_Latn", streaming=True):
    extra_shuffle_args = {}
    if streaming:
        extra_shuffle_args = {"buffer_size": 1000}

    if lang == "eng_Latn":
        ds = load_dataset(
            "HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=streaming
        ).shuffle(seed=42, **extra_shuffle_args).take(10000)
    else:
        ds = load_dataset("HuggingFaceFW/fineweb-2", lang, split="test", streaming=streaming)

    return ds


def create_valid_ds(
        output_dir: str,
        dataset: str = "fineweb",
        lang: str = "et",
        streaming: bool = True
):
    if dataset == "culturax":
        logging.info(f"Using CulturaX dataset with lang {lang}")
        ds_train = get_culturax_ds(streaming=streaming, lang=lang)
    elif dataset == "fineweb":
        logging.info(f"Using FineWeb dataset with lang {lang}")
        ds_train = get_fineweb_ds(streaming=streaming, lang=lang)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logging.info(f"Saving dataset to {output_dir}")
    ds = Dataset.from_list([{"text": doc["text"]} for doc in ds_train])
    ds.save_to_disk(output_dir)


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(create_valid_ds)
