from tokenizer_extension.sentencepiece_utils import train_sentencepiece_from_model

if __name__ == "__main__":
    import fire
    fire.Fire(train_sentencepiece_from_model)