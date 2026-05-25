from __future__ import annotations

from src.training.config import LoraConfig


def load_base_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool = True,
):
    """Load the base instruction model without attaching trainable adapters."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )


def attach_lora_for_training(model, config: LoraConfig, seed: int):
    """Attach LoRA adapters using Unsloth's optimized PEFT wrapper."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias=config.bias,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=seed,
    )


def prepare_for_inference(model) -> None:
    """Enable Unsloth inference kernels when available."""
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
