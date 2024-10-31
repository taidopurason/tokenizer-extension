import json
from itertools import islice

from tokenizers import Tokenizer

from .utils import update_postprocessor_special_tokens, get_vocab_and_merges, get_ordered_vocab, get_added_tokens_vocab


def prune_tokenizer(
        tokenizer, prune_ordered_tokens, n, ignore_added=True, ignore_special=True, ignore_tokens=None, verbose=False
):
    cfg = json.loads(tokenizer._tokenizer.to_str())
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    vocab, merges = get_vocab_and_merges(tokenizer)

    ignore_vocab = {}
    if ignore_tokens is not None:
        ignore_vocab.update({t: full_vocab[t] for t in ignore_tokens})

    if ignore_special or ignore_added:
        if not ignore_special:
            raise ValueError(
                "Added tokens can only be ignored along with special tokens, please set ignore_special=False")
        ignore_vocab.update(get_added_tokens_vocab(tokenizer, special_only=not ignore_added))

    tokens_to_prune = set(islice([x for x in prune_ordered_tokens if x not in ignore_vocab], n))

    cfg["model"]["merges"] = [" ".join(m) for m in merges if
                              all(t not in tokens_to_prune for t in m) and "".join(m) not in tokens_to_prune]
    new_vocab_tokens = [k for k in get_ordered_vocab(full_vocab) if k not in tokens_to_prune]
    new_full_vocab = {k: i for i, k in enumerate(new_vocab_tokens)}
    cfg["model"]["vocab"] = {k: new_full_vocab[k] for k in vocab if k in new_full_vocab}

    cfg["added_tokens"] = [
        dict(token, id=new_full_vocab[token["content"]])
        for token in cfg["added_tokens"] if token["content"] in new_full_vocab
    ]

    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(cfg))
    tokenizer = update_postprocessor_special_tokens(tokenizer)

    if verbose:
        new_full_vocab = tokenizer._tokenizer.get_vocab(True)
        new_reverse_vocab = {v: k for k, v in new_full_vocab.items()}
        for k, v in sorted(full_vocab.items(), key=lambda x: x[1]):
            new_id = new_full_vocab.get(k, None)
            if new_id is None:
                print(f"Removed {k} (id={v})")
            elif v != new_id:
                print(f"Moved {k} ({v} -> {new_id})")

        print("Empty indices:", [i for i in range(len(new_reverse_vocab)) if i not in new_reverse_vocab])

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
