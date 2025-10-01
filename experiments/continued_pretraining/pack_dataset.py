import logging
import os
import sys
from typing import List, Optional

import psutil
import numpy as np
from torch.utils.data import IterableDataset
import datasets
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, Features, Sequence, Value

def data_generator(iterator, features=("input_ids", "labels")):
    for x in iterator:
        yield {feature: x[feature] for feature in features}


def group_texts(
        input_ids: List[List[int]],
        labels: Optional[List[List[int]]] = None,
        sequence_length: int = 2048,
        concat_token_id: Optional[int] = None,
        add_position_ids: bool = False,
) -> dict:
    position_ids = None
    if concat_token_id is None:
        concatenated_input_ids = np.concatenate(input_ids)
        if labels is not None:
            concatenated_labels = np.concatenate(labels)
        if add_position_ids:
            position_ids = np.concatenate([list(range(len(x))) for x in input_ids])
    else:
        concatenated_input_ids = np.concatenate([x + [concat_token_id] for x in input_ids])
        if labels is not None:
            concatenated_labels = np.concatenate([x + [concat_token_id] for x in labels])
        if add_position_ids:
            position_ids = np.concatenate([list(range(len(x) + 1)) for x in input_ids])
    total_length = len(concatenated_input_ids)

    extra_fields = {}
    if add_position_ids:
        assert position_ids is not None
        extra_fields["position_ids"] = [
            position_ids[i: i + sequence_length] for i in
            range(0, total_length - sequence_length + 1, sequence_length)
        ]

    if labels is not None:
        extra_fields["labels"] = [
            concatenated_labels[i: i + sequence_length] for i in
            range(0, total_length - sequence_length + 1, sequence_length)
        ]

    return {
        "input_ids": [
            concatenated_input_ids[i: i + sequence_length] for i in
            range(0, total_length - sequence_length + 1, sequence_length)
        ],
        **extra_fields,
    }


def pack_dataset_fast(ds, tokenizer, seq_length, num_proc=-1, append_concat_token=True, add_position_ids=False,
                      add_labels=False):
    if num_proc == -1:
        num_proc = psutil.cpu_count()

    concat_token_id = getattr(tokenizer, 'eos_token_id', None)
    if concat_token_id is None and append_concat_token:
        raise ValueError("concat_token_id is not present")
    if not append_concat_token:
        logging.warning("Concat token will not be added")
        concat_token_id = None

    extra_features = {}
    if add_position_ids:
        extra_features["position_ids"] = Sequence(feature=Value(dtype="int64"), length=seq_length)

    extra_fields = []
    if add_labels:
        extra_fields.append("labels")
        extra_features["labels"] = Sequence(feature=Value(dtype="int64"), length=seq_length)

    return ds.map(
        lambda x, *y: group_texts(x, *y, concat_token_id=concat_token_id, sequence_length=seq_length,
                                  add_position_ids=add_position_ids),
        input_columns=["input_ids", *extra_fields],
        remove_columns=ds.column_names,
        batched=True,
        features=Features({
            "input_ids": Sequence(feature=Value(dtype="int64"), length=seq_length),
            **extra_features
        }),
        num_proc=num_proc,
    )


def main(
        tokenizer_name: str,
        dataset_name: str,
        output_dir: str,
        seq_length: int,
        local_dataset: bool = True,
        dataset_split: str = "train",
        append_concat_token: bool = True,
        max_in_memory_size: Optional[int] = None,
        shuffle: bool = False,
        limit_examples: Optional[int] = None,
        seed: int = 42,
        workers: int = -1,
        add_position_ids: bool = False,
        add_labels: bool = False,
):
    if max_in_memory_size is not None:
        datasets.config.IN_MEMORY_MAX_SIZE = max_in_memory_size
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if local_dataset:
        ds = load_from_disk(dataset_name)
    else:
        ds = load_dataset(dataset_name, split=dataset_split)

    logging.info(f"Loaded dataset {dataset_name} with {len(ds)} examples.")

    if shuffle:
        logging.info(f"Shuffling the dataset.")
        ds = ds.shuffle(seed=seed)

    if limit_examples is not None:
        ds = ds.take(limit_examples)

    logging.info(f"Starting packing dataset with {len(ds)} examples saving to {output_dir}.")

    packed_dataset = pack_dataset_fast(
        ds, tokenizer, seq_length,
        append_concat_token=append_concat_token, num_proc=workers, add_position_ids=add_position_ids,
        add_labels=add_labels,
    )
    logging.info(f"Example: {packed_dataset[0]}")
    packed_dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(main)
