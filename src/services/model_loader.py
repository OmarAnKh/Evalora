from functools import lru_cache
from typing import Tuple

DEFAULT_MODEL_NAME = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LOAD_IN_4BIT = True


@lru_cache(maxsize=None)
def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
) -> Tuple[object, object]:
    """Load a base model and tokenizer once and reuse them across the pipeline."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )


def get_model(
    model_name: str = DEFAULT_MODEL_NAME,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
) -> object:
    """Return a cached base model instance for the given configuration."""
    return load_model_and_tokenizer(model_name, max_seq_length, load_in_4bit)[0]


def get_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
) -> object:
    """Return a cached tokenizer instance for the given configuration."""
    return load_model_and_tokenizer(model_name, max_seq_length, load_in_4bit)[1]
