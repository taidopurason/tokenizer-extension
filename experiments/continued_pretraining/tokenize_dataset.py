import datasets
from datasets import load_dataset
import psutil
from transformers import AutoTokenizer
from typing import Optional


def tokenize_dataset(ds, tokenizer, num_proc=-1, text_col="text"):
    if num_proc == -1:
        num_proc = psutil.cpu_count()

    remove_cols = [x for x in ds.features.keys() if x not in ['input_ids', 'attention_mask']]
    return ds.map(
        lambda x: tokenizer(x[text_col]),
        batched=True,
        remove_columns=remove_cols,
        num_proc=num_proc,
        writer_batch_size=100,
        batch_size=100,
    )


def data_generator(iterator):
    yield from iterator


def main(
        tokenizer_name: str,
        dataset_name: str,
        output_dir: str,
        dataset_split: str = "train",
        text_col: str = "text",
        limit: Optional[int] = None,
        shuffle: bool = False,
        streaming: bool = False,
        shuffle_buffer: Optional[int] = None,
        seed: int = 42,
        workers: int = -1,
        max_in_memory_size: Optional[int] = None,
        config_name: Optional[str] = None,
        trust_remote_code: bool = False,
        force_redownload: bool = False,
        local_dataset: bool = False,
):
    if max_in_memory_size is not None:
        datasets.config.IN_MEMORY_MAX_SIZE = max_in_memory_size

    download_mode = None
    if force_redownload:
        download_mode = datasets.DownloadMode.FORCE_REDOWNLOAD

    if local_dataset:
        ds = datasets.load_from_disk(dataset_name)
    else:
        ds = load_dataset(
            dataset_name, name=config_name, split=dataset_split, streaming=streaming,
            trust_remote_code=trust_remote_code,
            download_mode=download_mode,
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if shuffle:
        print(f"Shuffling (buffer_size={shuffle_buffer})")
        opt_kwargs = {}
        if shuffle_buffer is not None:
            opt_kwargs["buffer_size"] = shuffle_buffer
        ds = ds.shuffle(**opt_kwargs, seed=seed)

    if limit is not None:
        print(f"Limiting dataset to {limit} examples")
        ds = ds.take(limit)

    if streaming:
        ds = datasets.Dataset.from_generator(
            data_generator, gen_kwargs={"iterator": ds}, features=ds.features
        )

    print(f"Tokenizing dataset with {len(ds)} examples")

    tokenized_dataset = tokenize_dataset(ds, tokenizer, text_col=text_col, num_proc=workers)
    tokenized_dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
