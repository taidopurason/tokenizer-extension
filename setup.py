from setuptools import setup, find_packages

setup(
    name="tokenizer_extension",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.44.0",
        "datasets>=2.21.0",
        "tiktoken>=0.7.0",
        "conllu",
        "numpy",
        "dask",
        "pandas",
        "tqdm",
        "requests",
        "nltk",
    ],
)