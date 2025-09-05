import json
from typing import List, Optional, Iterable, Dict
import logging

from tokenizers import Tokenizer
from tqdm import tqdm

from tokenizer_extension.utils import update_postprocessor_special_tokens, get_vocab_and_merges, get_ordered_vocab, \
    get_added_tokens_vocab

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


def _prune_tokenizer(tokenizer, tokens_to_prune: Iterable[str]):
    tokens_to_prune = set(tokens_to_prune)
    cfg = json.loads(tokenizer._tokenizer.to_str())
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    vocab, merges = get_vocab_and_merges(tokenizer)

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


class Pruner:
    def get_raw_pruning_order(self, tokenizer) -> List[str]:
        raise NotImplementedError()

    def get_pruning_order(
            self,
            tokenizer,
            ignore_added: bool = True,
            ignore_special: bool = True,
            ignore_tokens: Optional[List[str]] = None
    ) -> List[str]:
        return filter_special_tokens(
            tokenizer,
            self.get_raw_pruning_order(tokenizer),
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
        )[:n]

        if n is not None and len(tokens_to_prune) < n:
            raise ValueError(f"Not enough tokens to prune, {len(tokens_to_prune)} < {n}")

        _prune_tokenizer(
            tokenizer=tokenizer,
            tokens_to_prune=tokens_to_prune,
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

    def get_raw_pruning_order(self, tokenizer) -> List[str]:
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

    def get_raw_pruning_order(self, tokenizer) -> List[str]:
        return self.prune_ordered_tokens


class LastNPruner(Pruner):
    def get_raw_pruning_order(self, tokenizer) -> List[str]:
        full_vocab = tokenizer._tokenizer.get_vocab(True)
        return list(reversed(get_ordered_vocab(full_vocab)))


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
