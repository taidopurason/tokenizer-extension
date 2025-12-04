import logging
import os
import sys
from pathlib import Path
from typing import List, Set, Optional

import pandas as pd
from transformers import AutoTokenizer
from typing import List, Dict

from tokenizer_extension.benchmarking import evaluate_tokenizer, evaluate_tokenizer_self, evaluate_renyi_efficiency

from tokenizer_extension.data import load_flores as _load_flores
from datasets import load_dataset

from tokenizer_extension.utils import override_ignore_merges

flores_lang_map = {
    "ekk_Latn": "est_Latn",
    "fas_Arab": "pes_Arab",
    "cmn_Hani": "zho_Hans",
}

FINEWEB_CACHE = {}



class SentencePieceWrapper:
    def __init__(self, model_name: str, add_bos_token: bool = False, add_eos_token: bool = False):
        import sentencepiece as spm
        self.tokenizer = spm.SentencePieceProcessor(model_file=model_name)
        self.whitespace_symbol = "▁"

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.vocab = {self.tokenizer.id_to_piece(idx): idx for idx in range(self.tokenizer.vocab_size())}
        self.unk_token_id = self.tokenizer.unk_id()
        self.unk_token = self.tokenizer.id_to_piece(self.unk_token_id)

        self.bos_token_id = self.tokenizer.bos_id()
        self.bos_token = self.tokenizer.id_to_piece(self.bos_token_id)

        self.eos_token_id = self.tokenizer.eos_id()
        self.eos_token = self.tokenizer.id_to_piece(self.eos_token_id)

    def tokenize(self, text: str, add_special_tokens: bool = False) -> List[str]:
        return self.tokenizer.encode(text, out_type=str, add_bos=add_special_tokens and self.add_bos_token,
                                     add_eos=add_special_tokens and self.add_eos_token)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, out_type=int, add_bos=self.add_bos_token, add_eos=self.add_eos_token)

    def __call__(self, text: str) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text)}

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def __len__(self) -> int:
        return self.tokenizer.vocab_size()


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


def load_eval_datasets(lang, heldout=False, mixed_flores=False):
    extra = {}
    if heldout:
        extra[f"{lang}_fineweb_heldout"] = load_fineweb_heldout_data(lang)
    if mixed_flores:
        extra["mixed_flores"] = load_flores("eng_Latn") + load_flores(lang)
    return {
        f"{lang}_flores": load_flores(lang),
        "eng_Latn_flores": load_flores("eng_Latn"),
        **extra,
    }


def run_benchmark(
        tokenizer, lang: str, extension_vocab: Set[str] = None, is_sentencepiece: bool = False, heldout: bool = True,
        evaluate_renyi: bool = True, ignore_merges: Optional[bool] = None, return_frequencies: bool = False,
        mixed_flores: bool = False,
):
    self_eval_results = evaluate_tokenizer_self(tokenizer, extension_vocab)

    full_results = []
    with override_ignore_merges(tokenizer, ignore_merges):
        for name, dss in load_eval_datasets(lang, heldout, mixed_flores=mixed_flores).items():
            results = evaluate_tokenizer(
                tokenizer, dss, extension_vocab=extension_vocab, is_sentencepiece=is_sentencepiece,
                return_frequencies=return_frequencies
            )
            if evaluate_renyi:
                results["renyi_entropy"] = evaluate_renyi_efficiency(tokenizer, dss)
            full_results.append({
                "dataset": name,
                **self_eval_results,
                **results,
            })
    return full_results

def run_benchmark_sp(
        tokenizer, lang: str, extension_vocab: Set[str] = None, heldout: bool = True,
        evaluate_renyi: bool = True, ignore_merges: Optional[bool] = None, return_frequencies: bool = False,
        mixed_flores: bool = False,
):
    if evaluate_renyi:
        raise NotImplementedError("Renyi evaluation not implemented for SP tokenizers")
    if ignore_merges is not None:
        raise NotImplementedError("ignore_merges not applicable for SP tokenizers")

    full_results = []
    for name, dss in load_eval_datasets(lang, heldout, mixed_flores=mixed_flores).items():
        results = evaluate_tokenizer(
            tokenizer, dss, extension_vocab=extension_vocab, is_sentencepiece=True,
            return_frequencies=return_frequencies
        )

        full_results.append({
            "dataset": name,
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
        mixed_flores: bool = False,
        base_model_path: Optional[str] = None,
        implementation: str = "huggingface",
):
    if extension_values is None:
        extension_values = [0, 1000, 2000, 4000, 8000, 16000, 32000]

    if base_model_path is None:
        hf_model_name = model_dict[model_name]
    else:
        hf_model_name = base_model_path

    if implementation == "huggingface":
        base_model = AutoTokenizer.from_pretrained(hf_model_name)
        baseline_score = run_benchmark(
            base_model, lang=lang, extension_vocab=None, is_sentencepiece=is_sentencepiece, heldout=run_heldout_eval,
            evaluate_renyi=evaluate_renyi, ignore_merges=ignore_merges, return_frequencies=return_frequencies,
            mixed_flores=mixed_flores,
        )
    elif implementation == "sentencepiece":
        base_model = SentencePieceWrapper(hf_model_name)
        baseline_score = run_benchmark_sp(
            base_model, lang=lang, extension_vocab=None, heldout=run_heldout_eval,
            evaluate_renyi=evaluate_renyi, ignore_merges=ignore_merges, return_frequencies=return_frequencies,
            mixed_flores=mixed_flores,
        )
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    base_vocab_tokens = set(base_model.get_vocab())

    all_results = []

    for extension_method in ["continued-training", "retraining"]:
        for n_extension in extension_values:
            if n_extension == 0:
                results = baseline_score
            else:
                if implementation == "huggingface":
                    tokenizer_path = f"{experiment_path}/{dataset_name}-{lang}/tokenizers/{extension_method_map[extension_method]}-{model_name}-{budget}-ext{n_extension}"
                    extended_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    ext_model_tokens = set(extended_tokenizer.get_vocab())
                    ext_vocab = ext_model_tokens - base_vocab_tokens
                    assert len(ext_vocab) == n_extension, f"Expected {n_extension} extension tokens, got {len(ext_vocab)}"
                    results = run_benchmark(extended_tokenizer, lang=lang, extension_vocab=ext_vocab,
                                            is_sentencepiece=is_sentencepiece, heldout=run_heldout_eval,
                                            evaluate_renyi=evaluate_renyi, ignore_merges=ignore_merges,
                                            return_frequencies=return_frequencies,
                                            mixed_flores=mixed_flores,
                                            )
                elif implementation == "sentencepiece":
                    tokenizer_path = f"{experiment_path}/{dataset_name}-{lang}/sp-tokenizers/{extension_method_map[extension_method]}-{model_name}-{budget}-ext{n_extension}/tokenizer.model"
                    extended_tokenizer = SentencePieceWrapper(tokenizer_path)
                    ext_model_tokens = set(extended_tokenizer.get_vocab())
                    ext_vocab = ext_model_tokens - base_vocab_tokens
                    assert len(ext_vocab) == n_extension, f"Expected {n_extension} extension tokens, got {len(ext_vocab)}"
                    results = run_benchmark_sp(extended_tokenizer, lang=lang, extension_vocab=ext_vocab,
                                            heldout=run_heldout_eval,
                                            evaluate_renyi=evaluate_renyi, ignore_merges=ignore_merges,
                                            return_frequencies=return_frequencies,
                                            mixed_flores=mixed_flores,
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
