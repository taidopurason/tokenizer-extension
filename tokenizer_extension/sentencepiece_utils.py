from typing import Optional

try:
    import icu
except ImportError:
    icu = None


def get_script(char: str, prev: Optional[str] = None) -> str:
    if icu is None:
        raise ImportError('This function requires PyICU.')
    script = icu.Script.getScript(char).getName()
    if script == 'Hiragana' or script == 'Katakana':
        return 'Han'
    if script == 'Inherited' and prev is not None:
        return prev
    return script


START_SYMBOL = '▁'
SPECIAL = {'<unk>', '<s>', '</s>', '<pad>', '<mask>'}
BYTE_VOCAB = {'<0x00>', '<0x01>', '<0x02>', '<0x03>', '<0x04>', '<0x05>', '<0x06>', '<0x07>', '<0x08>', '<0x09>',
              '<0x0A>', '<0x0B>', '<0x0C>', '<0x0D>', '<0x0E>', '<0x0F>', '<0x10>', '<0x11>', '<0x12>', '<0x13>',
              '<0x14>', '<0x15>', '<0x16>', '<0x17>', '<0x18>', '<0x19>', '<0x1A>', '<0x1B>', '<0x1C>', '<0x1D>',
              '<0x1E>', '<0x1F>', '<0x20>', '<0x21>', '<0x22>', '<0x23>', '<0x24>', '<0x25>', '<0x26>', '<0x27>',
              '<0x28>', '<0x29>', '<0x2A>', '<0x2B>', '<0x2C>', '<0x2D>', '<0x2E>', '<0x2F>', '<0x30>', '<0x31>',
              '<0x32>', '<0x33>', '<0x34>', '<0x35>', '<0x36>', '<0x37>', '<0x38>', '<0x39>', '<0x3A>', '<0x3B>',
              '<0x3C>', '<0x3D>', '<0x3E>', '<0x3F>', '<0x40>', '<0x41>', '<0x42>', '<0x43>', '<0x44>', '<0x45>',
              '<0x46>', '<0x47>', '<0x48>', '<0x49>', '<0x4A>', '<0x4B>', '<0x4C>', '<0x4D>', '<0x4E>', '<0x4F>',
              '<0x50>', '<0x51>', '<0x52>', '<0x53>', '<0x54>', '<0x55>', '<0x56>', '<0x57>', '<0x58>', '<0x59>',
              '<0x5A>', '<0x5B>', '<0x5C>', '<0x5D>', '<0x5E>', '<0x5F>', '<0x60>', '<0x61>', '<0x62>', '<0x63>',
              '<0x64>', '<0x65>', '<0x66>', '<0x67>', '<0x68>', '<0x69>', '<0x6A>', '<0x6B>', '<0x6C>', '<0x6D>',
              '<0x6E>', '<0x6F>', '<0x70>', '<0x71>', '<0x72>', '<0x73>', '<0x74>', '<0x75>', '<0x76>', '<0x77>',
              '<0x78>', '<0x79>', '<0x7A>', '<0x7B>', '<0x7C>', '<0x7D>', '<0x7E>', '<0x7F>', '<0x80>', '<0x81>',
              '<0x82>', '<0x83>', '<0x84>', '<0x85>', '<0x86>', '<0x87>', '<0x88>', '<0x89>', '<0x8A>', '<0x8B>',
              '<0x8C>', '<0x8D>', '<0x8E>', '<0x8F>', '<0x90>', '<0x91>', '<0x92>', '<0x93>', '<0x94>', '<0x95>',
              '<0x96>', '<0x97>', '<0x98>', '<0x99>', '<0x9A>', '<0x9B>', '<0x9C>', '<0x9D>', '<0x9E>', '<0x9F>',
              '<0xA0>', '<0xA1>', '<0xA2>', '<0xA3>', '<0xA4>', '<0xA5>', '<0xA6>', '<0xA7>', '<0xA8>', '<0xA9>',
              '<0xAA>', '<0xAB>', '<0xAC>', '<0xAD>', '<0xAE>', '<0xAF>', '<0xB0>', '<0xB1>', '<0xB2>', '<0xB3>',
              '<0xB4>', '<0xB5>', '<0xB6>', '<0xB7>', '<0xB8>', '<0xB9>', '<0xBA>', '<0xBB>', '<0xBC>', '<0xBD>',
              '<0xBE>', '<0xBF>', '<0xC0>', '<0xC1>', '<0xC2>', '<0xC3>', '<0xC4>', '<0xC5>', '<0xC6>', '<0xC7>',
              '<0xC8>', '<0xC9>', '<0xCA>', '<0xCB>', '<0xCC>', '<0xCD>', '<0xCE>', '<0xCF>', '<0xD0>', '<0xD1>',
              '<0xD2>', '<0xD3>', '<0xD4>', '<0xD5>', '<0xD6>', '<0xD7>', '<0xD8>', '<0xD9>', '<0xDA>', '<0xDB>',
              '<0xDC>', '<0xDD>', '<0xDE>', '<0xDF>', '<0xE0>', '<0xE1>', '<0xE2>', '<0xE3>', '<0xE4>', '<0xE5>',
              '<0xE6>', '<0xE7>', '<0xE8>', '<0xE9>', '<0xEA>', '<0xEB>', '<0xEC>', '<0xED>', '<0xEE>', '<0xEF>',
              '<0xF0>', '<0xF1>', '<0xF2>', '<0xF3>', '<0xF4>', '<0xF5>', '<0xF6>', '<0xF7>', '<0xF8>', '<0xF9>',
              '<0xFA>', '<0xFB>', '<0xFC>', '<0xFD>', '<0xFE>', '<0xFF>'}
NUMBERS = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
}


def get_token_script(token: str, prev: Optional[str] = None):
    if len(token) == 0:
        raise ValueError("Empty token.")

    if len(token) == 1:
        return get_script(token, prev)

    if token[0] == START_SYMBOL:
        start = 1
    else:
        start = 0

    script = get_script(token[start], prev)
    for x in token[start + 1:]:
        current = get_script(x, prev=script)
        if script != current:
            print("Different scripts in one token", token)
        script = current

    return script


def group_tokens(text, tokenizer, separate_numbers=True, byte_fallback=True, special_tokens=None):
    separate_tokens = set()
    if special_tokens is None:
        special_tokens = SPECIAL
    separate_tokens.update(special_tokens)

    if byte_fallback:
        separate_tokens.update(BYTE_VOCAB)

    if separate_numbers:
        separate_tokens.update(NUMBERS)

    new_words = tokenizer.tokenize(text)
    grouped_new_words = []
    group = []

    prev_script = None

    for x in new_words:
        if x in separate_tokens:
            if len(group) > 0:
                grouped_new_words.append(group)
                group = []
            grouped_new_words.append([x])
            prev_script = None
            continue

        token_script = get_token_script(x, prev=prev_script)

        if x.startswith(START_SYMBOL) or (prev_script is not None and token_script != prev_script):
            if len(group) > 0:
                grouped_new_words.append(group)
                group = []
        group.append(x)

        prev_script = token_script

    if len(group) > 0:
        grouped_new_words.append(group)

    return list(map(tuple, grouped_new_words))


def read_model(path: str):
    from sentencepiece.sentencepiece_model_pb2 import ModelProto
    model = ModelProto()
    with open(path, "rb") as f:
        model.ParseFromString(f.read())
    return model


def train_sentencepiece_from_model(
        pretraind_model_path: str,
        input: str,
        vocab_size: int = 64000,
        model_prefix: str = "sp_model",
        num_threads: int = 16,
):
    import sentencepiece as spm

    model = read_model(pretraind_model_path)
    config = {x[0].name: getattr(model.trainer_spec, x[0].name) for x in model.trainer_spec.ListFields()}
    assert config["model_type"] == 2
    kwargs = {
        k: config[k]
        for k in [
            'character_coverage', 'input_sentence_size', 'seed_sentencepiece_size', 'shrinking_factor',
            'num_sub_iterations', 'max_sentence_length', 'shuffle_input_sentence', 'max_sentencepiece_length',
            'split_by_unicode_script', 'split_by_whitespace', 'split_by_number', 'treat_whitespace_as_suffix',
            'split_digits', 'allow_whitespace_only_pieces', 'vocabulary_output_piece_score', 'hard_vocab_limit',
            'use_all_vocab', 'byte_fallback'
        ]
    }

    print(f"Training SentencePiece model with the following parameters: {kwargs}", flush=True)

    spm.SentencePieceTrainer.train(
        input=input,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        num_threads=num_threads,
        model_type="bpe",
        **kwargs
    )
