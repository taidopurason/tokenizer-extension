import json
from collections import defaultdict
from itertools import islice
from typing import List, Optional, Iterable, Dict
import logging

from tokenizers import Tokenizer
from tqdm import tqdm

from tokenizer_extension.iterative_tokenizer import IterativeTokenizer
from tokenizer_extension.utils import update_postprocessor_special_tokens, get_vocab_and_merges, get_ordered_vocab, \
    get_added_tokens_vocab

try:
    import icu
except ImportError:
    icu = None

logger = logging.getLogger("tokenizer_extension.pruning")


def filter_special_tokens(
        tokenizer,
        tokens: List[str],
        ignore_added: bool = True,
        ignore_special: bool = True,
        ignore_tokens: List[str] = None,
) -> List[str]:
    full_vocab = tokenizer._tokenizer.get_vocab(True)

    ignore_vocab = {}
    if ignore_tokens is not None:
        ignore_vocab.update({t: full_vocab[t] for t in ignore_tokens})

    if ignore_special or ignore_added:
        ignore_vocab.update({
            x: full_vocab[x]
            for x in [tokenizer.unk_token, tokenizer.eos_token, tokenizer.bos_token, tokenizer.pad_token]
            if x is not None
        })

        if not ignore_special:
            raise ValueError(
                "Added tokens can only be ignored along with special tokens, please set ignore_special=True"
            )
        ignore_vocab.update(get_added_tokens_vocab(tokenizer, special_only=not ignore_added))

    return [x for x in tokens if x not in ignore_vocab]


def prune_tokenizer(
        tokenizer,
        prune_ordered_tokens: List[str],
        n: Optional[int] = None,
        ignore_added: bool = True,
        ignore_special: bool = True,
        ignore_tokens: List[str] = None,
):
    cfg = json.loads(tokenizer._tokenizer.to_str())
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    vocab, merges = get_vocab_and_merges(tokenizer)

    tokens_to_prune = set(islice(filter_special_tokens(
        tokenizer,
        prune_ordered_tokens,
        ignore_added=ignore_added,
        ignore_special=ignore_special,
        ignore_tokens=ignore_tokens
    ), n))

    if n is not None and len(tokens_to_prune) < n:
        raise ValueError(f"Not enough tokens to prune, {len(tokens_to_prune)} < {n}")

    cfg["model"]["merges"] = [
        " ".join(m) for m in merges
        if all(t not in tokens_to_prune for t in m) and "".join(m) not in tokens_to_prune
    ]
    new_vocab_tokens = [k for k in get_ordered_vocab(full_vocab) if k not in tokens_to_prune]
    new_full_vocab = {k: i for i, k in enumerate(new_vocab_tokens)}
    cfg["model"]["vocab"] = {k: new_full_vocab[k] for k in vocab if k in new_full_vocab}

    cfg["added_tokens"] = [
        dict(token, id=new_full_vocab[token["content"]])
        for token in cfg["added_tokens"] if token["content"] in new_full_vocab
    ]

    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(cfg))
    tokenizer = update_postprocessor_special_tokens(tokenizer)

    new_full_vocab = tokenizer._tokenizer.get_vocab(True)
    new_reverse_vocab = {v: k for k, v in new_full_vocab.items()}
    for k, v in sorted(full_vocab.items(), key=lambda x: x[1]):
        new_id = new_full_vocab.get(k, None)
        if new_id is None:
            logger.debug(f"Removed {k} (id={v})")
        elif v != new_id:
            logger.debug(f"Moved {k} ({v} -> {new_id})")

    logger.debug(f"Empty indices: {[i for i in range(len(new_reverse_vocab)) if i not in new_reverse_vocab]}")

    return tokenizer


def prune_tokenizer_last(tokenizer, n, ignore_added=True, ignore_special=True, ignore_tokens=None, verbose=False):
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    reverse_ordered_vocab = list(reversed(get_ordered_vocab(full_vocab)))
    return prune_tokenizer(
        tokenizer,
        reverse_ordered_vocab,
        n,
        ignore_added=ignore_added,
        ignore_special=ignore_special,
        ignore_tokens=ignore_tokens,
        verbose=verbose
    )


class Pruner:
    def _get_pruning_order(self, tokenizer) -> List[str]:
        raise NotImplementedError()

    def get_pruning_order(
            self,
            tokenizer,
            ignore_added: bool = True,
            ignore_special: bool = True,
            ignore_tokens: Optional[List[str]] = None
    ):
        return filter_special_tokens(
            tokenizer,
            self._get_pruning_order(tokenizer),
            ignore_added=ignore_added,
            ignore_special=ignore_special,
            ignore_tokens=ignore_tokens,
        )

    def prune(
            self,
            tokenizer,
            n: Optional[int] = None,
            ignore_added: bool = True,
            ignore_special: bool = True,
            ignore_tokens: Optional[List[str]] = None
    ):
        tokens_to_prune = self.get_pruning_order(
            tokenizer, ignore_added=ignore_added, ignore_special=ignore_special, ignore_tokens=ignore_tokens
        )
        prune_tokenizer(
            tokenizer=tokenizer,
            prune_ordered_tokens=tokens_to_prune,
            n=n,
            ignore_added=False,
            ignore_special=False,
        )
        return tokenizer


class TrainablePruner(Pruner):
    def __init__(self):
        self.is_trained = False
        self.prune_ordered_tokens = []

    def _train_pruning_order(self, tokenizer, training_data: List[str]) -> List[str]:
        raise NotImplementedError()

    def train(self, tokenizer, training_data: List[str]):
        self.prune_ordered_tokens = self._train_pruning_order(tokenizer, training_data)
        self.is_trained = True
        return self

    def _get_pruning_order(self, tokenizer) -> List[str]:
        if not self.is_trained:
            raise ValueError("Pruner is not trained yet")
        return self.prune_ordered_tokens

    def train_prune(
            self,
            tokenizer,
            training_data: List[str],
            n: Optional[int],
            ignore_added: bool = True,
            ignore_special: bool = True,
            ignore_tokens: Optional[List[str]] = None
    ):
        self.train(tokenizer, training_data)
        return self.prune(
            tokenizer=tokenizer,
            n=n,
            ignore_added=ignore_added,
            ignore_special=ignore_special,
            ignore_tokens=ignore_tokens
        )


class PretrainedPruner(Pruner):
    def __init__(self, prune_ordered_tokens: List[str]):
        super().__init__()
        self.prune_ordered_tokens = prune_ordered_tokens

    def _get_pruning_order(self, tokenizer) -> List[str]:
        return self.prune_ordered_tokens


class LastNPruner(Pruner):
    def _get_pruning_order(self, tokenizer) -> List[str]:
        full_vocab = tokenizer._tokenizer.get_vocab(True)
        return list(reversed(get_ordered_vocab(full_vocab)))


class ScriptPruner(Pruner):
    def __init__(
            self,
            allowed_scripts: Optional[Iterable[str]] = None,
            forbidden_scripts: Optional[Iterable[str]] = None
    ):
        super().__init__()
        if icu is None:
            raise ImportError("icu module is required for script pruning")
        self.allowed_scripts = set(allowed_scripts) if allowed_scripts is not None else None
        self.forbidden_scripts = set(forbidden_scripts) if forbidden_scripts is not None else None

    @staticmethod
    def icu_script_filter(text: str, allowed_scripts=None, forbidden_scripts=None) -> bool:
        scripts = [icu.Script.getScript(x).getName() for x in text]

        if allowed_scripts is not None and not all([script in allowed_scripts for script in scripts]):
            return False

        if forbidden_scripts is not None and any([script in forbidden_scripts for script in scripts]):
            return False

        return True

    def _get_pruning_order(self, tokenizer) -> List[str]:
        vocab = tokenizer.get_vocab()
        tokens_to_remove = [
            x for x in reversed(get_ordered_vocab(vocab))
            if not self.icu_script_filter(
                tokenizer.decode(vocab[x]),
                allowed_scripts=self.allowed_scripts,
                forbidden_scripts=self.forbidden_scripts
            )
        ]
        return tokens_to_remove


class LatinScriptPruner(ScriptPruner):
    def __init__(self):
        super().__init__(allowed_scripts={"Latin", "Common", "Inherited"})


class LatinCyrillicScriptPruner(ScriptPruner):
    def __init__(self):
        super().__init__(allowed_scripts={"Cyrillic", "Latin", "Common", "Inherited"})


def calculate_token_frequency(tokenizer, texts) -> Dict[str, int]:
    token_freqs = {t: 0 for t in tokenizer._tokenizer.get_vocab(True)}
    for text in tqdm(texts, miniters=len(texts) // 100):
        for token in tokenizer.tokenize(text):
            token_freqs[token] += 1
    return token_freqs


def calculate_frequency_order(tokenizer, texts) -> List[str]:
    token_counts = calculate_token_frequency(tokenizer=tokenizer, texts=texts)
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    return [tok for tok, _ in sorted(token_counts.items(), key=lambda x: (x[1], -full_vocab[x[0]]))]


class FrequencyPruner(TrainablePruner):
    def _train_pruning_order(self, tokenizer, training_data: List[str]) -> List[str]:
        return calculate_frequency_order(tokenizer=tokenizer, texts=training_data)


def calculate_merge_statistics(tokenizer, texts, ignore_merges=None):
    vocab, merges = get_vocab_and_merges(tokenizer)
    if ignore_merges is None:
        ignore_merges = tokenizer._tokenizer.model.ignore_merges

    step_tokenizer = IterativeTokenizer(
        vocab, merges, ignore_merges=ignore_merges, byte_fallback=tokenizer._tokenizer.model.byte_fallback,
        unk_token=tokenizer._tokenizer.model.unk_token
    )

    merge_counts = {m: 0 for m in merges}
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    token_counts = {t: 0 for t in full_vocab}

    pre_tokenizer = tokenizer._tokenizer.pre_tokenizer
    normalizer = tokenizer._tokenizer.normalizer
    normalize_func = normalizer.normalize_str if normalizer is not None else lambda x: x
    for text in tqdm(texts, miniters=len(texts) // 100):
        pre_tokenized_str = pre_tokenizer.pre_tokenize_str(normalize_func(text))
        for word, _ in pre_tokenized_str:
            step_tokens = None
            for step, merge in step_tokenizer.tokenize_iteratively(word):
                step_tokens = step_tokenizer.word_to_tokens(step)
                merge_result = step_tokenizer.merge_id_to_token(merge)
                if merge_result is not None:
                    merge_counts[merge_result[0]] += 1

            if step_tokens is None:
                raise ValueError(f"No tokens returned by tokenizer for word {word}")
            for t in step_tokens:
                token_counts[t] += 1

    return token_counts, merge_counts


def merge_based_pruning_order(vocab, token_counts, merge_counts):
    token_counts = defaultdict(int, token_counts)
    for m, freq in merge_counts.items():
        for t in m:
            token_counts[t] += freq

    return [tok for tok, _ in sorted(token_counts.items(), key=lambda x: (x[1], -vocab[x[0]]))]


class MergeBasedPruner(TrainablePruner):
    def __init__(self, ignore_merges: bool = None):
        super().__init__()
        self.ignore_merges = ignore_merges
        self._token_counts = None
        self._merge_counts = None

    def _train_pruning_order(self, tokenizer, training_data: List[str]) -> List[str]:
        full_vocab = tokenizer._tokenizer.get_vocab(True)
        token_counts, merge_counts = calculate_merge_statistics(
            tokenizer=tokenizer, texts=training_data, ignore_merges=self.ignore_merges
        )
        self._token_counts = token_counts
        self._merge_counts = merge_counts
        return merge_based_pruning_order(
            vocab=full_vocab, token_counts=token_counts, merge_counts=merge_counts
        )


def calculate_orders(
        tokenizer,
        texts,
        calculate_token_frequency=False,
        calculate_merge_based_pruning=False,
        ignore_merges=None,
        return_counts=False
):
    _, merges = get_vocab_and_merges(tokenizer)
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    orders = {
        "last_n": list(reversed(get_ordered_vocab(full_vocab)))
    }

    # another tokenization step on the whole dataset
    if calculate_token_frequency:
        logging.info("Calculating HF token frequency")
        orders["token_frequency"] = calculate_frequency_order(tokenizer=tokenizer, texts=texts)

    # tokenize the whole dataset
    if calculate_merge_based_pruning:
        logging.info("Calculating merge statistics")
        token_counts, merge_counts = calculate_merge_statistics(tokenizer=tokenizer, texts=texts,
                                                                ignore_merges=ignore_merges)
        logging.info("Calculating orders")
        orders["least_used_token"] = merge_based_pruning_order(
            vocab=full_vocab, token_counts=token_counts, merge_counts=merge_counts
        )
        orders["_token_frequency"] = [
            tok for tok, _ in sorted(token_counts.items(), key=lambda x: (x[1], -full_vocab[x[0]]))
        ]
        if return_counts:
            orders["_token_counts"] = token_counts
            orders["_merge_counts"] = merge_counts

    return orders
