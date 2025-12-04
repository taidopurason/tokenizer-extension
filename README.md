# Tokenizer Extension

This repository provides a lightweight toolkit for extending and pruning the vocabularies of pre-trained models that use BPE tokenizers. 
It primarily supports HuggingFace **byte-level BPE** tokenizers (e.g., Llama-3, Qwen-2.5), with partial support for **SentencePiece BPE** tokenizers (e.g., Llama-2) via both HuggingFace Transformers and the original SentencePiece implementation.

The toolkit implements **continued BPE training** for vocabulary extension as well as **vocabulary pruning** strategies presented in our paper:  
[**Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models**](https://arxiv.org/abs/2512.03989) (preprint).

**Usage Guide:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taidopurason/tokenizer-extension/blob/main/tokenizer_extension_example.ipynb)

## Installation
After cloning this repository it can be installed using pip:
```
pip install .
```

## Usage
This toolkit includes utilities for:
* Adding or removing tokens from a tokenizer’s vocabulary
* Identifying the most optimal tokens to add by continuing BPE training on a given text corpus
* Pruning the vocabulary using various strategies
* Resizing and initializing model embeddings accordingly (with support for Fast Vocabulary Transfer)
* Benchmarking and comparing tokenizers



### Training vocabulary extension (Continued BPE training)
💡 Continued BPE training provides an effective way to extend a tokenizer’s vocabulary for a new domain.  
By **continuing the original BPE training process** on a target text corpus, it learns new tokens and merges that remain fully compatible with the existing vocabulary. 
This ensures that all added tokens and merges are fully compatible and optimal under the BPE objective, without introducing any unreachable tokens.
As a result, continued BPE training often achieves better text compression for the same vocabulary size compared to adding tokens from an auxiliary tokenizer.


```
from transformers import AutoTokenizer
from tokenizer_extension.train_vocab_extension import train_vocab_extension
from tokenizer_extension.utils import write_json


tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH)
# replace with your data:
train_docs = ["This is a sample document.", "This is another document."]
extension_tokens = train_vocab_extension(
    tokenizer=tokenizer,
    corpus=train_docs,
    extension_size=64000,
    is_sentencepiece=False,
)

# saving
write_json(extension_tokens["vocab"], "vocab.json")
write_json([" ".join(x) for x in extension_tokens["merges"]], "merges.json")
```

See also scripts/train_vocab_extension.py for a command line tool.

### Extending the vocabulary
```
from transformers import AutoTokenizer
from tokenizer_extension.extension import extend_tokenizer
from tokenizer_extension.utils import read_json

new_vocab = read_json("vocab.json")
new_merges = [tuple(x.split(" ")) for x in read_json("merges.json")]

tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH)
tokenizer = extend_tokenizer(
    tokenizer,
    new_vocab=new_vocab,
    new_merges=new_merges,
    n_tokens=1000,
    keep_added_token_positions=False,
)
tokenizer.save_pretrained(output_path)
```

Generating merges based on the vocabulary:
```
from transformers import AutoTokenizer
from tokenizer_extension.extension import extend_tokenizer
from tokenizer_extension.utils import read_json

new_vocab = {"new_token1": 0, "new_token2": 1, "new_token3": 2}

tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH)
tokenizer = extend_tokenizer(
    tokenizer,
    new_vocab=new_vocab,
    new_merges=None,
    n_tokens=3,
    keep_added_token_positions=False,
)
tokenizer.save_pretrained(output_path)
```
Note the extension works in-place, changing the tokenizer object.
### Pruning the vocabulary
We provide various pruning strategies. 

Trainable with a corpus:
* Frequency
* **Frequency (leaf)**
* **Merge-based**

Do not need a training corpus:
* Last N (ID-based)
* **Last N (leaf)**
* Script-based

💡 We recommend using leaf-based pruning strategies as they tend to yield better performance without creating unreachable tokens.
Merge-based pruning yields similar results to leaf-based frequency pruning, also avoiding unreachable tokens.

Note that the pruning works in-place changing the tokenizer object.
Usage example:
```
from transformers import AutoTokenizer
from tokenizer_extension.pruning import LeafFrequencyPruner

tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH)
train_docs = ["This is a sample document.", "This is another document."]

pruner = LeafFrequencyPruner()
pruner.train(tokenizer, train_docs)

# saving the pruner
pruner.save("leaf_freq.json")

# pruning n tokens (in-place)
pruner.prune(tokenizer, n_tokens)
```

Pruner not requiring training corpus:
```
from transformers import AutoTokenizer
from tokenizer_extension.pruning import LastNPruner

tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH)

pruner = LastNPruner()

pruner.train(tokenizer) # or pruner.train(tokenizer, None) 
pruner.save("last_n.json")

# pruning n tokens (in-place)
pruner.prune(tokenizer, n_tokens)
```

Loading the saved pruner:
```
from transformers import AutoTokenizer
from tokenizer_extension.pruning import PretrainedPruner

n_tokens = 1000  # number of tokens to prune
pruner = PretrainedPruner.load("leaf_freq.json")
pruner.prune(AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH), n_tokens)
```

Note: Pruning SentencePiece models is currently not implemented.

### Modifying the model embeddings
Modifying the model embeddings to match the new tokenizer vocabulary.
This example uses Fast Vocabulary Transfer (mean of constituents) for initializing new embeddings.
```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizer_extension.models import modify_embeddings
from tokenizer_extension.utils import write_json

pretrained_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)
original_tokenizer = AutoTokenizer.from_pretrained(model_path)
new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_path)

changes = modify_embeddings(
    pretrained_model, 
    old_tokenizer=original_tokenizer, 
    new_tokenizer=new_tokenizer, 
    init_method="mean_of_constituents", 
    ignore_size_mismatch=False
)

out_path = "modified_model"
pretrained_model.save_pretrained(out_path)
new_tokenizer.save_pretrained(out_path)
write_json(changes, os.path.join(out_path, "embedding_changes.json"))
```

### Benchmarking the tokenizer
We provide tools for finding unreachable tokens (not reachable by merges) in the tokenizer.
```
    from tokenizer_extension.benchmarking import find_unreachable_tokens_tokenization
    unreachable_tokens_tok = find_unreachable_tokens_tokenization(tokenizer)
    n_unreachable = len(unreachable_tokens_tok)
```

### CLI
See scripts in the `scripts/` directory for command line usage of the tools.

## Citation
````
@misc{purason2025teachingoldtokenizersnew,
      title={Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models}, 
      author={Taido Purason and Pavel Chizhov and Ivan P. Yamshchikov and Mark Fishel},
      year={2025},
      eprint={2512.03989},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.03989}, 
}
````
