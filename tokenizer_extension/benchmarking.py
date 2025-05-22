from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool
from typing import Optional, Iterable, Dict, Tuple, List, Union, Set

import tiktoken
from tokenizers import pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from tokenizer_extension.utils import get_vocab_and_merges, get_added_tokens_vocab
from tokenizer_extension.sentencepiece_utils import BYTE_VOCAB

HF_BYTE_VOCAB = set(pre_tokenizers.ByteLevel.alphabet())
SP_BYTE_VOCAB = set(BYTE_VOCAB)

DEFAULT_WHITESPACE_SYMBOL = "Ġ"
SP_WHITESPACE_SYMBOL = "▁"

# from https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee
byte_encoder = bytes_to_unicode()


def token_bytes_to_string(b):
    return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])


class TiktokenTokenizerWrapper:
    def __init__(self, model_name: str):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.vocab = {token_bytes_to_string(k): v for k, v in self.tokenizer._mergeable_ranks.items()}
        self.vocab.update(self.tokenizer._special_tokens)
        self.unk_token = None
        self.unk_token_id = None

    def tokenize(self, text: str) -> List[str]:
        return list(map(token_bytes_to_string, self.tokenizer.decode_tokens_bytes(self.tokenizer.encode(text))))

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def __len__(self) -> int:
        return self.tokenizer.n_vocab


class SentencePieceWrapper:
    def __init__(self, model_name: str):
        import sentencepiece as spm
        self.tokenizer = spm.SentencePieceProcessor(model_file=model_name)
        self.vocab = {self.tokenizer.id_to_piece(idx): idx for idx in range(self.tokenizer.vocab_size())}
        self.unk_token = self.tokenizer.id_to_piece(self.tokenizer.unk_id())
        self.unk_token_id = self.tokenizer.unk_id()

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text, out_type=str)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def __len__(self) -> int:
        return self.tokenizer.vocab_size()


TOKENIZER_TYPE = Union[PreTrainedTokenizerFast, PreTrainedTokenizer, TiktokenTokenizerWrapper, SentencePieceWrapper]


@dataclass
class TokenizerConfig:
    name: str
    tokenizer: TOKENIZER_TYPE
    start_symbol: str = DEFAULT_WHITESPACE_SYMBOL
    evaluation_kwargs: dict = field(default_factory=dict)

    @classmethod
    def load_hf(
            cls,
            name,
            is_sentencepiece: bool = False,
            tokenizer_kwargs: Optional[dict] = None,
            evaluation_kwargs: Optional[dict] = None
    ):
        return cls(
            name=name,
            start_symbol=SP_WHITESPACE_SYMBOL if is_sentencepiece else DEFAULT_WHITESPACE_SYMBOL,
            tokenizer=AutoTokenizer.from_pretrained(name, **(evaluation_kwargs or dict())),
            evaluation_kwargs=(tokenizer_kwargs or dict())
        )

    @classmethod
    def load_tiktoken(cls, name, evaluation_kwargs: Optional[dict] = None):
        return cls(
            name=name,
            start_symbol=DEFAULT_WHITESPACE_SYMBOL,
            tokenizer=TiktokenTokenizerWrapper(name),
            evaluation_kwargs=(evaluation_kwargs or dict())
        )


def _calculate_metrics(
        vocab_size: int,
        total_tokens: int,
        total_chars: int,
        total_words: int,
        total_bytes: int,
        total_examples: int,
        unknown_tokens: int
) -> Dict[str, float]:
    return {
        "vocab_size": vocab_size,
        "dataset_size": total_examples,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_bytes": total_bytes,
        "unknown_tokens": unknown_tokens,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0,
        "tokens_per_char": total_tokens / total_chars if total_chars else 0,
        "tokens_per_byte": total_tokens / total_bytes if total_bytes else 0,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0,
        "words_per_token": total_words / total_tokens if total_tokens else 0,
        "tokens_per_word": total_tokens / total_words if total_words else 0,
        "tokens_per_sequence": total_tokens / total_examples if total_examples else 0,
        "unknown_tokens_per_tokens": unknown_tokens / total_tokens if total_tokens else 0,
    }


# A simple benchmark method
def evaluate_tokenizer_slow(
        tokenizer,
        data: Iterable[str],
        ignore_empty: bool = True,
        extension_vocab: Optional[Dict[str, int]] = None,
        byte_vocab: Set[str] = None,
        is_sentencepiece: bool = False
) -> Dict[str, float]:
    total_tokens = 0
    total_chars = 0
    total_words = 0
    total_examples = 0
    unknown_tokens = 0
    total_bytes = 0
    vocab = tokenizer.get_vocab()
    vocab_usage = {k: 0 for k in vocab}

    if byte_vocab is None:
        byte_vocab = SP_BYTE_VOCAB if is_sentencepiece else HF_BYTE_VOCAB

    total_byte_fallbacks = 0

    for text in data:
        if text == "":
            if ignore_empty:
                print("Empty text in dataset.")
                continue
            raise ValueError("Empty text in dataset.")

        tokens = tokenizer.tokenize(text)
        if byte_vocab is not None:
            total_byte_fallbacks += len([t for t in tokens if t in byte_vocab])

        for t in tokens:
            vocab_usage[t] += 1
        total_tokens += len(tokens)
        total_bytes += len(text.encode('utf-8'))
        total_chars += len(text)
        total_words += len(text.split())
        total_examples += 1
        unknown_tokens += len([t for t in tokens if t not in vocab])

    vocab_utilisation = len(set(k for k, v in vocab_usage.items() if v != 0)) / len(vocab)
    if extension_vocab is not None and len(extension_vocab) > 0:
        extension_utilisation = len(set(k for k in extension_vocab if vocab_usage[k] != 0)) / len(extension_vocab)
        extensions_tokens = sum(vocab_usage[k] for k in extension_vocab)
    else:
        extension_utilisation = 0
        extensions_tokens = 0

    return {
        **_calculate_metrics(
            vocab_size=len(vocab),
            total_tokens=total_tokens,
            total_chars=total_chars,
            total_words=total_words,
            total_bytes=total_bytes,
            total_examples=total_examples,
            unknown_tokens=unknown_tokens
        ),
        "extension_usage_rate": extension_utilisation,
        "vocab_usage_rate": vocab_utilisation,
        "total_extension_tokens": extensions_tokens,
        "extension_tokens_per_token": extensions_tokens / total_tokens if total_tokens else 0,
        "byte_fallback_rate": total_byte_fallbacks / total_tokens if total_tokens else 0,
    }


def _process_text(text: str, tokenizer, ignore_empty: bool, vocab) -> Tuple[int, int, int, int, int, int]:
    if text == "":
        if ignore_empty:
            print("Empty text in dataset.")
            return (0, 0, 0, 0, 0, 0)
        raise ValueError("Empty text in dataset.")

    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    num_bytes = len(text.encode("utf-8"))
    num_chars = len(text)
    num_words = len(text.split())
    num_unknown = len([t for t in tokens if t not in vocab])

    return (1, num_tokens, num_chars, num_words, num_bytes, num_unknown)


def evaluate_tokenizer(tokenizer: TOKENIZER_TYPE, data: Iterable[str], ignore_empty: bool = True) -> Dict[str, float]:
    vocab = tokenizer.get_vocab()

    def worker(text):
        return _process_text(text, tokenizer, ignore_empty, vocab)

    with ThreadPool() as pool:
        results = pool.map(worker, data)

    total_examples, total_tokens, total_chars, total_words, total_bytes, unknown_tokens = map(sum, zip(*results))

    return _calculate_metrics(
        vocab_size=len(vocab),
        total_tokens=total_tokens,
        total_chars=total_chars,
        total_words=total_words,
        total_bytes=total_bytes,
        total_examples=total_examples,
        unknown_tokens=unknown_tokens
    )


def find_unreachable_tokens_tokenization(tokenizer, ignore_added=True):
    original_value = None
    try:
        vocab, merges = get_vocab_and_merges(tokenizer)
        if hasattr(tokenizer._tokenizer.model, "ignore_merges"):
            original_value = tokenizer._tokenizer.model.ignore_merges
            tokenizer._tokenizer.model.ignore_merges = False

        reachability = {}
        for tok in vocab:
            values = [t.value for t in tokenizer._tokenizer.model.tokenize(tok)]
            reachability[tok] = len(values) == 1 and values[0] == tok

        unreachable_tokens = set(tok for tok, value in reachability.items() if value == False)
        if ignore_added:
            unreachable_tokens = unreachable_tokens - set(get_added_tokens_vocab(tokenizer))
        return unreachable_tokens
    finally:
        if original_value is not None:
            tokenizer._tokenizer.model.ignore_merges = original_value


def fill_reachability(token, reachability, token_merges, alphabet):
    if token in reachability:
        return

    if token in alphabet:
        reachability[token] = True
        return

    if token not in token_merges:
        reachability[token] = False
        return

    reachability[token] = False
    for t1, t2 in token_merges[token]:
        fill_reachability(t1, reachability, token_merges, alphabet)
        fill_reachability(t2, reachability, token_merges, alphabet)
        if reachability[t1] and reachability[t2]:
            reachability[token] = True
            return


def find_unreachable_tokens(tokenizer, is_sentencepiece=False, ignore_added=True):
    vocab, merges = get_vocab_and_merges(tokenizer)
    alphabet = set(x for x in vocab if len(x) == 1) if is_sentencepiece else set(pre_tokenizers.ByteLevel.alphabet())
    token_merges = {}
    for token in vocab:
        token_merges[token] = []

    for merge in merges:
        token_merges["".join(merge)].append(merge)

    reachability = {}
    for token in vocab:
        fill_reachability(token, reachability, token_merges, alphabet)

    unreachable_tokens = set(tok for tok, value in reachability.items() if value == False)

    if ignore_added:
        unreachable_tokens = unreachable_tokens - set(get_added_tokens_vocab(tokenizer))

    return unreachable_tokens
