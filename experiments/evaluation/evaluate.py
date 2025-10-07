import logging
import os
import sys
from pathlib import Path
from typing import List, Set, Optional

import pandas as pd
from transformers import AutoTokenizer

from tokenizer_extension.benchmarking import evaluate_tokenizer, evaluate_tokenizer_self, evaluate_renyi_entropy

from tokenizer_extension.data import load_flores as _load_flores
from datasets import load_dataset

from tokenizer_extension.utils import override_ignore_merges

flores_lang_map = {
    "ekk_Latn": "est_Latn",
    "fas_Arab": "pes_Arab",
    "cmn_Hani": "zho_Hans",
}

FINEWEB_CACHE = {}


def get_fineweb_ds(lang="ekk_Latn"):
    if lang == "eng_Latn":
        ds = load_dataset(
            "HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True
        ).shuffle(seed=42, buffer_size=1000).take(10000)
    else:
        ds = load_dataset("HuggingFaceFW/fineweb-2", lang, split="test", streaming=True).shuffle(buffer_size=1000)

    return ds


def load_fineweb_heldout_data(lang, limit=10000):
    if lang in FINEWEB_CACHE:
        return FINEWEB_CACHE[lang]

    ds = [x["text"] for x in get_fineweb_ds(lang)]
    if limit is not None:
        ds = ds[:limit]

    FINEWEB_CACHE[lang] = ds
    return ds


FLORES_CACHE = {}


def load_flores(lang):
    if lang in FLORES_CACHE:
        return FLORES_CACHE[lang]
    ds = _load_flores(flores_lang_map[lang] if lang in flores_lang_map else lang)
    FLORES_CACHE[lang] = ds
    return ds


def load_eval_datasets(lang, heldout=False):
    extra = {}
    if heldout:
        extra[f"{lang}_fineweb_heldout"] = load_fineweb_heldout_data(lang)
    return {
        f"{lang}_flores": load_flores(lang),
        "eng_Latn_flores": load_flores("eng_Latn"),
        **extra,
    }


def run_benchmark(
        tokenizer, lang: str, extension_vocab: Set[str] = None, is_sentencepiece: bool = False, heldout: bool = True,
        evaluate_renyi: bool = True, ignore_merges: Optional[bool] = None, return_frequencies: bool = False
):
    self_eval_results = evaluate_tokenizer_self(tokenizer, extension_vocab)

    full_results = []
    with override_ignore_merges(tokenizer, ignore_merges):
        for name, dss in load_eval_datasets(lang, heldout).items():
            results = evaluate_tokenizer(
                tokenizer, dss, extension_vocab=extension_vocab, is_sentencepiece=is_sentencepiece,
                return_frequencies=return_frequencies
            )
            if evaluate_renyi:
                results["renyi_entropy"] = evaluate_renyi_entropy(tokenizer, dss)
            full_results.append({
                "dataset": name,
                **self_eval_results,
                **results,
            })
    return full_results


model_dict = {
    "llama3": "meta-llama/Meta-Llama-3.1-8B",
    "llama2": "meta-llama/Llama-2-7b-hf",
    "mistralnemo": "mistralai/Mistral-Nemo-Base-2407",
    "qwen25": "Qwen/Qwen2.5-3B-Instruct",
}

extension_method_map = {
    "continued-training": "extension",
    "retraining": "fromscratch",
}


def eval_language(
        lang: str,
        model_name: str,
        is_sentencepiece: bool,
        experiment_path: str,
        out_dir: str = "scores",
        dataset_name: str = "fineweb",
        budget: int = 1000000000,
        extension_values: Optional[List[int]] = None,
        run_heldout_eval: bool = True,
        evaluate_renyi: bool = True,
        ignore_merges: Optional[bool] = None,
        return_frequencies: bool = False,
):
    if extension_values is None:
        extension_values = [0, 1000, 2000, 4000, 8000, 16000, 32000]

    hf_model_name = model_dict[model_name]

    base_model = AutoTokenizer.from_pretrained(hf_model_name)
    baseline_score = run_benchmark(
        base_model, lang=lang, extension_vocab=None, is_sentencepiece=is_sentencepiece, heldout=run_heldout_eval,
        evaluate_renyi=evaluate_renyi, ignore_merges=ignore_merges, return_frequencies=return_frequencies
    )

    base_vocab_tokens = set(base_model.get_vocab())

    all_results = []

    for extension_method in ["continued-training", "retraining"]:
        for n_extension in extension_values:
            if n_extension == 0:
                results = baseline_score
            else:

                tokenizer_path = f"{experiment_path}/{dataset_name}-{lang}/tokenizers/{extension_method_map[extension_method]}-{model_name}-{budget}-ext{n_extension}"
                extended_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                ext_model_tokens = set(extended_tokenizer.get_vocab())
                ext_vocab = ext_model_tokens - base_vocab_tokens
                assert len(ext_vocab) == n_extension, f"Expected {n_extension} extension tokens, got {len(ext_vocab)}"
                results = run_benchmark(extended_tokenizer, lang=lang, extension_vocab=ext_vocab,
                                        is_sentencepiece=is_sentencepiece, heldout=run_heldout_eval,
                                        evaluate_renyi=evaluate_renyi, ignore_merges=ignore_merges
                                        )
            results = [
                {
                    "tokenizer_name": f"{model_name}-{budget}-{extension_method}-{n_extension}",
                    "size": res["vocab_size"],
                    "base_tokenizer": model_name,
                    "n_extension": n_extension,
                    "extension_method": extension_method,
                    **res
                }
                for res in results
            ]
            all_results.extend(results)

    df_base = pd.DataFrame(baseline_score)
    df_all = pd.DataFrame(all_results)
    df_ct = df_all[df_all["extension_method"] == "continued-training"]
    df_rt = df_all[df_all["extension_method"] == "retraining"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df_base.to_csv(f"{out_dir}/{lang}_{model_name}_baseline.csv")
    df_ct.to_csv(f"{out_dir}/{lang}_{model_name}_continued-training.csv")
    df_rt.to_csv(f"{out_dir}/{lang}_{model_name}_retrain.csv")


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(eval_language)
