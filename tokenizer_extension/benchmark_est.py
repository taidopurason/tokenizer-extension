from tqdm import tqdm

from .benchmarking import evaluate_tokenizer
from .data import load_flores, load_starcoder, load_enc


def benchmark(tokenizer, tokenizer_name=None, metrics=None, eval_datasets=None):
    result = {"tokenizer": tokenizer_name}
    for name, dataset in eval_datasets.items():
        res = evaluate_tokenizer(tokenizer, tqdm(dataset, desc=f"Evaluating {tokenizer_name} on {name}"))
        for col in (list(res.keys()) if metrics is None else metrics):
            result[f"{name}_{col}"] = res[col]
    return result


def load_est_data_small():
    return {
        "et_flores_devtest": load_flores("est_Latn"),
        "en_flores_devtest": load_flores("eng_Latn"),
        "starcoder_1k": load_starcoder(n=1000),
        "enc_valid_1k": load_enc(n=1000),
    }


def load_small_est_benchmark(metrics=("chars_per_token", "tokens_per_word")):
    data = load_est_data_small()

    def run_benchmark(tokenizer, tokenizer_name=None):
        return benchmark(tokenizer, tokenizer_name, metrics=metrics, eval_datasets=data)

    return run_benchmark
