import logging
from itertools import islice
from typing import Optional, List, Tuple, Dict, Union
from tokenizers import pre_tokenizers
from transformers.convert_slow_tokenizer import generate_merges

from .utils import get_ordered_vocab, get_vocab_and_merges, replace_tokenizer_vocab_merges, get_added_tokens_vocab, \
    update_postprocessor_special_tokens


def get_vocab_scores(
        vocab: Union[List[str], Dict[str, int]],
        alphabet: Optional[List[str]] = None,
        added_tokens: Optional[List[str]] = None
) -> List[Tuple[str, int]]:
    if alphabet is None:
        alphabet = []

    if added_tokens is None:
        added_tokens = []

    if isinstance(vocab, dict):
        vocab = get_ordered_vocab(vocab)

    ignore_vocab = set(alphabet + added_tokens)

    score = -1
    scores = []
    for t in vocab:
        if t in ignore_vocab:
            scores.append(0)
        else:
            scores.append(score)
            score -= 1

    return [(t, s) for t, s in zip(vocab, scores)]


def extend_vocab(
        new_vocab: Dict[str, int],
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        max_token_id: int,
        new_merges: Optional[List[Tuple[str, str]]] = None,
        n_tokens: Optional[int] = None,
        generate_new_merges: bool = False,
        alphabet: Optional[List[str]] = None,
        added_tokens: Optional[List[str]] = None,
        prepend_merges: bool = False,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    if alphabet is None:
        alphabet = list(sorted(pre_tokenizers.ByteLevel.alphabet()))

    new_vocab_filtered = {k: v for k, v in new_vocab.items() if k not in vocab and k not in added_tokens}
    if n_tokens is not None and len(new_vocab_filtered) < n_tokens:
        raise ValueError(
            f"Not enough new tokens to add. Found {len(new_vocab_filtered)}, but expected {n_tokens}."
        )

    new_vocab_filtered = {
        k: max_token_id + v + 1
        for v, k in enumerate(islice(get_ordered_vocab(new_vocab_filtered), n_tokens))
    }
    combined_vocab = {**vocab, **new_vocab_filtered}

    if generate_new_merges:
        combined_vocab_scores = get_vocab_scores(combined_vocab, alphabet=alphabet, added_tokens=added_tokens)
        combined_merges = generate_merges(combined_vocab, combined_vocab_scores)
        return combined_vocab, combined_merges

    if new_merges is None:
        new_vocab_scores = get_vocab_scores(new_vocab_filtered, alphabet=alphabet, added_tokens=added_tokens)
        new_merges = generate_merges(combined_vocab, new_vocab_scores)
    else:
        new_merges = [
            (a, b) for a, b in new_merges if a in combined_vocab and b in combined_vocab and a + b in combined_vocab
        ]

    combined_merges = new_merges + merges if prepend_merges else merges + new_merges

    return combined_vocab, combined_merges


def extend_tokenizer(
        tokenizer,
        new_vocab: Dict[str, int],
        new_merges: Optional[List[Tuple[str, str]]] = None,
        n_tokens: int = None,
        generate_new_merges: bool = False,
        prepend_merges: bool = False,
        alphabet: List[str] = None,
        keep_added_token_positions: bool = True
):
    vocab, merges = get_vocab_and_merges(tokenizer)
    max_token_id = max(v for _, v in tokenizer._tokenizer.get_vocab(keep_added_token_positions).items())
    added_tokens_vocab = get_added_tokens_vocab(tokenizer)
    ext_vocab, ext_merges = extend_vocab(
        new_vocab,
        vocab,
        merges,
        max_token_id,
        new_merges=new_merges,
        n_tokens=n_tokens,
        generate_new_merges=generate_new_merges,
        added_tokens=list(get_added_tokens_vocab(tokenizer).keys()),
        prepend_merges=prepend_merges,
        alphabet=alphabet
    )
    ext_vocab_reverse = {idx: token for token, idx in ext_vocab.items()}
    if keep_added_token_positions:
        logging.info("Adding added tokens to the vocabulary if not present to preserve their indices")
        for token, idx in added_tokens_vocab.items():
            if token in ext_vocab:
                continue
            if idx in ext_vocab_reverse:
                raise ValueError("Added token and extended vocabulary have tokens with the same indices")
            ext_vocab[token] = idx
    tokenizer = replace_tokenizer_vocab_merges(tokenizer, ext_vocab, ext_merges)
    if not keep_added_token_positions:
        tokenizer = update_postprocessor_special_tokens(tokenizer)
    if len(set(tokenizer.get_vocab().keys())) != len(set(tokenizer.get_vocab().values())):
        raise ValueError("Tokens with the same ID found in vocabulary.")
    return tokenizer
