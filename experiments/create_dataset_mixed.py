import logging
import os
import random
import sys
from typing import List

from datasets import load_dataset, Dataset

from tokenizer_extension.utils import budget_iterator


def get_culturax_ds(lang="et", streaming=True):
    extra_shuffle_args = {}
    if streaming:
        extra_shuffle_args = {"buffer_size": 1000}

    return load_dataset(
        "uonlp/CulturaX", lang, split="train", streaming=streaming
    ).shuffle(seed=42, **extra_shuffle_args).skip(10000)


def get_fineweb_ds(lang="ekk_Latn", streaming=True):
    extra_shuffle_args = {}
    if streaming:
        extra_shuffle_args = {"buffer_size": 1000}

    if lang == "eng_Latn":
        ds = load_dataset(
            "HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=streaming
        ).shuffle(seed=42, **extra_shuffle_args).skip(10000)
    else:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-2", lang, split="train", streaming=streaming
        ).shuffle(seed=42, **extra_shuffle_args)
    return ds



def get_codeparrot_ds(lang="all-all", streaming=True):
    extra_shuffle_args = {}
    if streaming:
        extra_shuffle_args = {"buffer_size": 1000}

    ds = load_dataset(
        "codeparrot/github-code-clean", lang, split="train", streaming=streaming
    ).rename_column("code", "text")

    return ds.shuffle(seed=42, **extra_shuffle_args).skip(10000)

def get_ds(dataset, lang, streaming):
    if dataset == "culturax":
        logging.info(f"Using CulturaX dataset with lang {lang}")
        ds_train = get_culturax_ds(streaming=streaming, lang=lang)
    elif dataset == "fineweb":
        logging.info(f"Using FineWeb dataset with lang {lang}")
        ds_train = get_fineweb_ds(streaming=streaming, lang=lang)
    elif dataset == "codeparrot":
        logging.info(f"Using CodeParrot dataset with lang {lang}")
        ds_train = get_codeparrot_ds(streaming=streaming, lang=lang)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return ds_train


def create_budget_ds(
        output_dir: str,
        langs: List[str],
        weights: List[float],
        streaming: List[bool],
        datasets: List[str],
        char_budget: int = 1_000_000_000,
):
    if isinstance(langs, str):
        langs = list(langs.split(","))
    logging.info(f"Creating budget dataset with {char_budget} characters for langs {langs} ({datasets})")

    weights = [w / sum(weights) for w in weights]
    budgets = [int(char_budget * w) for w in weights]

    if len(datasets) == 1:
        datasets = [datasets[0]] * len(langs)
    assert len(datasets) == len(langs), "Datasets and langs must have the same length"
    assert len(langs) == len(budgets), "Langs and weights must have the same length"
    training_data = []
    for lang, budget, stream, dataset in zip(langs, budgets, streaming, datasets):
        logging.info(f"Using {budget} characters for lang {lang}")
        ds_train = get_ds(dataset, lang, stream)
        train_docs = [{"text": doc, "lang": lang} for doc in budget_iterator(ds_train, budget)]
        logging.info(f"Got {len(train_docs)} documents for lang {lang}")
        training_data.extend(train_docs)

    if len(langs) > 1:
        logging.info("Shuffling training data")
        random.Random(42).shuffle(training_data)


    logging.info(f"Saving dataset to {output_dir}")
    ds = Dataset.from_list(training_data)
    ds.save_to_disk(output_dir)


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(create_budget_ds)
