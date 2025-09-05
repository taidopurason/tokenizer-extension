import logging
from typing import List, Optional

from .base import PretrainedPruner, LastNPruner, FrequencyPruner
from .merge_pruner import MergeBasedPruner

logger = logging.getLogger("tokenizer_extension.pruning")


# Legacy implementation
def calculate_orders(
        tokenizer,
        texts,
        calculate_token_frequency=False,
        calculate_merge_based_pruning=False,
        ignore_merges=None,
        return_counts=False
):
    orders = {
        "last_n": LastNPruner().get_raw_pruning_order(tokenizer=tokenizer)
    }

    # another tokenization step on the whole dataset
    if calculate_token_frequency:
        logging.info("Calculating HF token frequency")
        orders["token_frequency"] = FrequencyPruner().train(tokenizer, texts).get_raw_pruning_order(tokenizer=tokenizer)

    # tokenize the whole dataset
    if calculate_merge_based_pruning:
        logging.info("Calculating merge order")
        merge_pruner = MergeBasedPruner(ignore_merges=ignore_merges).train(tokenizer, texts)
        orders["least_used_token"] = merge_pruner.get_raw_pruning_order(tokenizer=tokenizer)
        if return_counts:
            orders["_token_counts"] = merge_pruner._token_counts
            orders["_merge_counts"] = merge_pruner._merge_counts

    return orders


# Legacy implementation
def prune_tokenizer(
        tokenizer,
        prune_ordered_tokens: List[str],
        n: Optional[int] = None,
        ignore_added: bool = True,
        ignore_special: bool = True,
        ignore_tokens: List[str] = None,
):
    return PretrainedPruner(prune_ordered_tokens=prune_ordered_tokens).prune(
        tokenizer=tokenizer, n=n, ignore_added=ignore_added, ignore_special=ignore_special, ignore_tokens=ignore_tokens
    )
