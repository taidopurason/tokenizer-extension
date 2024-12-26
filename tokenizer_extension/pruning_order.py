import logging
from collections import defaultdict
from typing import List, Dict

from tqdm import tqdm

from tokenizer_extension.iterative_tokenizer import IterativeTokenizer
from tokenizer_extension.utils import get_vocab_and_merges, get_ordered_vocab


def calculate_token_counts(data, tokenizer) -> Dict[str, int]:
    token_freqs = {t: 0 for t in tokenizer._tokenizer.get_vocab(True)}
    for text in tqdm(data, miniters=len(data) // 100):
        for token in tokenizer.tokenize(text):
            token_freqs[token] += 1
    return token_freqs


def calculate_frequency_order(train_data, tokenizer) -> List[str]:
    token_counts = calculate_token_counts(train_data, tokenizer)
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    return [tok for tok, _ in sorted(token_counts.items(), key=lambda x: (x[1], -full_vocab[x[0]]))]


def calculate_merge_statistics(texts, tokenizer, ignore_merges=None):
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


def least_used_token_pruning_order(vocab, token_counts, merge_counts):
    token_counts = defaultdict(int, token_counts)
    for m, freq in merge_counts.items():
        for t in m:
            token_counts[t] += freq

    return [tok for tok, _ in sorted(token_counts.items(), key=lambda x: (x[1], -vocab[x[0]]))]


def merge_pruning_order(vocab, merges, merge_counts):
    if merges is None:
        merges = list(merge_counts.keys())

    token_merges = {t: set() for t in vocab}
    for m in merges:
        token_merges["".join(m)].add(m)

    tokens_to_remove = []
    for merge, _ in sorted(merge_counts.items(), key=lambda x: x[1]):
        token = "".join(merge)

        token_merges[token].remove(merge)
        if len(token_merges[token]) == 0:
            tokens_to_remove.append(token)
            del token_merges[token]

    return tokens_to_remove


# For comparing different pruning orders, we can use the following code:
def calculate_orders(texts, tokenizer, ignore_merges=None, calculate_hf_impl=False, return_counts=False):
    _, merges = get_vocab_and_merges(tokenizer)
    full_vocab = tokenizer._tokenizer.get_vocab(True)

    # tokenize the whole dataset
    logging.info("Calculating merge statistics")
    token_counts, merge_counts = calculate_merge_statistics(texts, tokenizer, ignore_merges=ignore_merges)
    logging.info("Calculating orders")
    orders = {
        "least_used_token": least_used_token_pruning_order(
            vocab=full_vocab, token_counts=token_counts, merge_counts=merge_counts
        ),
        "merge": merge_pruning_order(
            vocab=full_vocab, merges=merges, merge_counts=merge_counts
        ),
        "token_frequency": [tok for tok, _ in sorted(token_counts.items(), key=lambda x: (x[1], -full_vocab[x[0]]))],
        "last_n": list(reversed(get_ordered_vocab(full_vocab))),
    }

    # another tokenization step on the whole dataset
    if calculate_hf_impl:
        logging.info("Calculating HF implementation")
        orders["hf_token_frequency"] = calculate_frequency_order(texts, tokenizer)

    # for debugging purposes
    if return_counts:
        orders["_token_counts"] = token_counts
        orders["_merge_counts"] = merge_counts

    return orders
