from transformers import AutoTokenizer

from tokenizer_extension.pruning import TrainablePrunerBase, StaticPrunerBase, register_pruner
from tokenizer_extension.utils import get_vocab_and_merges
from collections import defaultdict, Counter
from heapdict import heapdict
from typing import List, Dict, Optional


def compute_token_frequencies(
        tokenizer: AutoTokenizer,
        corpus: List[str],
) -> Dict[int, int]:
    counts = Counter()
    for text in corpus:
        counts.update(tokenizer.encode(text))
    return dict(counts)


def leaf_pruning_order_id(
        tokenizer: AutoTokenizer,
):
    vocab, merges = get_vocab_and_merges(tokenizer)
    leaves = set()
    atomics = set(vocab.values())
    token_splits = dict()
    downstream_merges = defaultdict(int)
    for left, right in merges:
        left_index, right_index = vocab[left], vocab[right]
        token_index = vocab[left + right]
        atomics.discard(token_index)
        leaves.discard(left_index)
        leaves.discard(right_index)
        leaves.add(token_index)
        token_splits[token_index] = [left_index, right_index]
        downstream_merges[left_index] += 1
        downstream_merges[right_index] += 1
    pruning_order = []
    for i in sorted(vocab.values(), reverse=True):
        if i in atomics or i not in leaves: continue
        left, right = token_splits[i]
        downstream_merges[left] -= 1
        if not downstream_merges[left]:
            leaves.add(left)
        downstream_merges[right] -= 1
        if not downstream_merges[right]:
            leaves.add(right)
        pruning_order.append(i)
    return pruning_order


def leaf_pruning_order_frequency(
        tokenizer: AutoTokenizer,
        corpus: List[str],
):
    vocab, merges = get_vocab_and_merges(tokenizer)
    frequencies = compute_token_frequencies(tokenizer, corpus)
    frequencies = defaultdict(int, frequencies)
    leaves = set()
    atomics = set(vocab.values())
    token_splits = dict()
    downstream_merges = defaultdict(int)
    for left, right in merges:
        left_index, right_index = vocab[left], vocab[right]
        token_index = vocab[left + right]
        atomics.discard(token_index)
        leaves.discard(left_index)
        leaves.discard(right_index)
        leaves.add(token_index)
        token_splits[token_index] = [left_index, right_index]
        downstream_merges[left_index] += 1
        downstream_merges[right_index] += 1
    queue = heapdict()
    for token in leaves:
        queue[token] = frequencies[token]
    pruning_order = []
    while queue:
        token, freq = queue.popitem()
        if token in atomics or token not in leaves: continue
        left, right = token_splits[token]
        frequencies[left] += frequencies[token]
        frequencies[right] += frequencies[token]
        frequencies[token] = 0
        downstream_merges[left] -= 1
        if not downstream_merges[left]:
            leaves.add(left)
            queue[left] = frequencies[left]
        downstream_merges[right] -= 1
        if not downstream_merges[right]:
            leaves.add(right)
            queue[right] = frequencies[right]
        pruning_order.append(token)
    return pruning_order


def convert_ids_to_tokens(tokenizer, ids: List[int]) -> List[str]:
    reverse_vocab = {idx: tok for tok, idx in tokenizer.get_vocab().items()}
    return [reverse_vocab[x] for x in ids]


@register_pruner("leaf_frequency")
class LeafFrequencyPruner(TrainablePrunerBase):
    def calculate_pruning_order(self, tokenizer, training_data: List[str]) -> List[str]:
        return convert_ids_to_tokens(tokenizer, leaf_pruning_order_frequency(tokenizer, training_data))


@register_pruner("leaf_last_n")
class LeafLastNPruner(StaticPrunerBase):
    def calculate_pruning_order(self, tokenizer, training_data: Optional[List[str]] = None) -> List[str]:
        return convert_ids_to_tokens(tokenizer, leaf_pruning_order_id(tokenizer))
