from omegaconf import DictConfig


def build_llm(cfg: DictConfig):
    """
    Load a base LLM via Unsloth and attach a LoRA adapter.

    Returns (model, tokenizer) ready for SFTTrainer.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        dtype=None,                        # auto: bf16 on Ampere+, fp16 otherwise
        load_in_4bit=cfg.model.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.model.lora_r,
        target_modules=list(cfg.model.lora_target_modules),
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
