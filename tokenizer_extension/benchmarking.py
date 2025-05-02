from dataclasses import dataclass, field
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer
import tiktoken
from typing import List, Union
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from typing import Dict, Optional
from typing import Iterable, Dict

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
def evaluate_tokenizer_slow(tokenizer: TOKENIZER_TYPE, data: Iterable[str], ignore_empty: bool = True) -> Dict[
    str, float]:
    total_tokens = 0
    total_chars = 0
    total_words = 0
    total_examples = 0
    unknown_tokens = 0
    total_bytes = 0
    vocab = tokenizer.get_vocab()

    for text in data:
        if text == "":
            if ignore_empty:
                print("Empty text in dataset.")
                continue
            raise ValueError("Empty text in dataset.")

        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)
        total_bytes += len(text.encode('utf-8'))
        total_chars += len(text)
        total_words += len(text.split())
        total_examples += 1
        unknown_tokens += len([t for t in tokens if t not in vocab])
    return _calculate_metrics(
        vocab_size=len(vocab),
        total_tokens=total_tokens,
        total_chars=total_chars,
        total_words=total_words,
        total_bytes=total_bytes,
        total_examples=total_examples,
        unknown_tokens=unknown_tokens
    )


from multiprocessing.pool import ThreadPool
from typing import Iterable, Dict, Tuple


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
