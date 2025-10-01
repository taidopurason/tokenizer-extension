import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

import datasets
import torch
from accelerate import PartialState
from datasets import load_dataset, load_from_disk
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer, \
    PreTrainedTokenizer, PreTrainedModel, Trainer, TrainingArguments, default_data_collator

from torch.utils.data import Dataset

from trl.trainer import ConstantLengthDataset

HF_DATASET_TYPE = "huggingface"
LOCAL_PACKED_HF_DATASET_TYPE = "huggingface_local_packed"
DATASET_TYPES = [HF_DATASET_TYPE, LOCAL_PACKED_HF_DATASET_TYPE]


@dataclass
class ScriptArguments:
    train_path: str = field(metadata={"help": "Training data path"})
    valid_path: Optional[str] = field(metadata={"help": "Validation data path"}, default=None)
    train_dataset_type: str = field(
        default=HF_DATASET_TYPE,
        metadata={"help": "Training dataset type", "choices": DATASET_TYPES}
    )
    train_split_name: str = field(default="train")
    valid_split_name: str = field(default="validation")
    valid_split_limit: Optional[int] = field(default=None)
    valid_dataset_type: str = field(
        default=HF_DATASET_TYPE,
        metadata={"help": "Training dataset type", "choices": DATASET_TYPES}
    )
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    torch_dtype: Optional[str] = field(default=None)
    low_cpu_mem_usage: bool = field(default=False)
    use_flash_attention_2: bool = field(default=False)
    save_final_model: bool = field(default=False)
    stream_train_dataset: bool = field(default=False)
    stream_valid_dataset: bool = field(default=False)
    allow_empty_pad_token: bool = field(default=False)
    pad_token: Optional[str] = field(default=None)
    max_seq_length: Optional[int] = None
    eval_packing: bool = field(default=False)
    dataset_text_field: str = field(default="text")
    examples_per_dataset: Optional[str] = field(default=None)


class HFLocalPackedDataset(Dataset):
    def __init__(
            self,
            paths: Union[str, List[str]],
            examples_per_dataset: Optional[List[int]] = None,
            seed: int = 42,
    ):
        if isinstance(paths, str):
            paths = paths.split(",")

        dss = []
        for dataset_name in paths:
            ds = load_from_disk(dataset_name)
            logging.info(f"Loaded dataset {dataset_name} with {len(ds)} examples.")
            dss.append(ds)

        if examples_per_dataset is not None:
            assert len(examples_per_dataset) == len(dss)
            dss = [
                self.sample_dataset(ds, target_size=n_examples, seed=seed)
                for ds, n_examples in zip(dss, examples_per_dataset)
            ]
            logging.info(f"Upsampled datasets to {[len(ds) for ds in dss]} examples each.")

        if len(dss) == 1:
            self.dataset = dss[0]
        else:
            self.dataset = datasets.concatenate_datasets(dss).shuffle(seed=seed)

        logging.info(f"Created final dataset with {len(self.dataset)} examples.")

    @staticmethod
    def sample_dataset(dataset: datasets.Dataset, target_size: int, seed: int = 42):
        if len(dataset) == target_size:
            return dataset

        dataset = dataset.shuffle(seed=seed)
        n_epochs = math.ceil(target_size / len(dataset))
        if n_epochs > 1:
            dataset = datasets.concatenate_datasets([dataset] * n_epochs)
        return dataset.take(target_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ds_object = self.dataset[idx]
        example = {
            "input_ids": torch.LongTensor(ds_object["input_ids"]),
            "labels": torch.LongTensor(ds_object.get("labels", ds_object["input_ids"])),
            **{k: torch.LongTensor(ds_object[k]) for k in ["attention_mask", "position_ids"] if k in ds_object}
        }

        return example


def create_dataset(
        path: str,
        tokenizer: PreTrainedTokenizer,
        args: ScriptArguments,
        training_args: TrainingArguments,
        split: str = "train",
        streaming: bool = False,
        dataset_type: str = HF_DATASET_TYPE,
):
    logging.info(f"Loading dataset from: {path}")
    if dataset_type == HF_DATASET_TYPE:
        if ":" in path:
            ds_name, lang = path.split(":")
            return load_dataset(ds_name, lang, streaming=streaming, split=split)
        return load_dataset(path, streaming=streaming, split=split)
    if dataset_type == LOCAL_PACKED_HF_DATASET_TYPE:
        if args.examples_per_dataset is not None:
            examples_per_dataset = [int(x) for x in args.examples_per_dataset.split(",")]
        else:
            examples_per_dataset = None

        return HFLocalPackedDataset(
            args.train_path,
            examples_per_dataset=examples_per_dataset
        )

    raise ValueError("Unsupported dataset type")


def create_train_dataset(tokenizer: PreTrainedTokenizer, args: ScriptArguments, training_args: TrainingArguments):
    return create_dataset(
        tokenizer=tokenizer, args=args, training_args=training_args, path=args.train_path, split=args.train_split_name,
        streaming=args.stream_train_dataset, dataset_type=args.train_dataset_type
    )


def data_generator(iterator):
    yield from iterator


def _create_valid_dataset(
        path: str,
        tokenizer,
        args: ScriptArguments,
        training_args: TrainingArguments,
):
    with PartialState().local_main_process_first():
        valid_ds = create_dataset(
            tokenizer=tokenizer, args=args, training_args=training_args, path=path, split=args.valid_split_name,
            streaming=args.stream_valid_dataset, dataset_type=args.valid_dataset_type
        )
        if args.valid_split_limit is not None:
            valid_ds = valid_ds.take(args.valid_split_limit)
        if args.eval_packing:
            valid_ds = ConstantLengthDataset(
                tokenizer,
                valid_ds,
                infinite=False,
                seq_length=args.max_seq_length,
                chars_per_token=3.6,
                dataset_text_field=args.dataset_text_field,
                shuffle=False,
            )
        if args.stream_valid_dataset:
            valid_ds = datasets.Dataset.from_generator(
                data_generator, gen_kwargs={"iterator": valid_ds}
            )
        return valid_ds


def create_valid_dataset(tokenizer: PreTrainedTokenizer, args: ScriptArguments, training_args: TrainingArguments):
    logging.info(f"Creating validation dataset from: {args.valid_path}")
    if args.valid_path is None:
        return None

    if len(args.valid_path.split(",")) == 1:
        return _create_valid_dataset(args.valid_path, tokenizer, args, training_args)

    valid_datasets = {}
    for path in args.valid_path.split(","):
        valid_name, *valid_path = path.split(":")
        valid_path = ":".join(valid_path)
        valid_datasets[valid_name] = _create_valid_dataset(valid_path, tokenizer, args, training_args)
    return valid_datasets


def create_and_prepare_model(
        args: ScriptArguments, training_args: TrainingArguments
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if args.torch_dtype is None or args.torch_dtype == "auto":
        torch_dtype = args.torch_dtype
    else:
        torch_dtype = getattr(torch, args.torch_dtype)
    logging.info(f"Using torch dtype: {torch_dtype}")

    model_kwargs = {}

    if args.use_flash_attention_2:
        logging.info("Using Flash Attention 2")
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=None,
        use_cache=not training_args.gradient_checkpointing,
        trust_remote_code=True,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        **model_kwargs,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name if args.tokenizer_name is None else args.tokenizer_name,
        model_max_length=args.max_seq_length,
        padding_side="right",
    )

    if args.pad_token is not None:
        logging.info(f"Setting pad token to {args.pad_token}")
        tokenizer.pad_token = args.pad_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model.padding_idx = tokenizer.pad_token_id
        model.model.embed_tokens.padding_idx = tokenizer.pad_token_id

    if tokenizer.pad_token is None and not args.allow_empty_pad_token:
        logging.warning("Pad token is not set, using EOS token as pad token.")
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Using pad token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
    logging.info(f"Using eos token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    return model, tokenizer


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(script_args: ScriptArguments, training_args: TrainingArguments):
    torch.cuda.manual_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)

    model, tokenizer = create_and_prepare_model(
        script_args, training_args
    )

    train_dataset = create_train_dataset(tokenizer, script_args, training_args)
    eval_dataset = create_valid_dataset(tokenizer, script_args, training_args)

    logging.info(f"Max sequence length: {tokenizer.model_max_length}")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.accelerator.print(f"{trainer.model}")
    print_trainable_parameters(trainer.model)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if script_args.save_final_model:
        final_out = os.path.join(training_args.output_dir, "last_checkpoint")
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            trainer.save_model(final_out)
        else:
            trainer.model.save_pretrained(final_out)
        tokenizer.save_pretrained(final_out)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    args, training_args = parser.parse_args_into_dataclasses()
    main(
        script_args=args,
        training_args=training_args,
    )
