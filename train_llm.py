import os
import sys
import wandb
from omegaconf import OmegaConf
from transformers import TrainingArguments
from trl import SFTTrainer

from src.models.llm import build_llm
from src.loaders.text_loader import build_text_dataset


def train(config_path: str = "configs/train_llm.yaml") -> None:
    cfg = OmegaConf.load(config_path)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity") or None,
        tags=list(cfg.wandb.get("tags", [])),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    model, tokenizer = build_llm(cfg)
    dataset = build_text_dataset(cfg, tokenizer=tokenizer)

    os.makedirs(cfg.checkpoint.dirpath, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=cfg.training.per_device_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            warmup_steps=cfg.training.warmup_steps,
            num_train_epochs=cfg.training.max_epochs,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            fp16=cfg.training.get("fp16", False),
            bf16=cfg.training.get("bf16", True),
            lr_scheduler_type=cfg.training.get("lr_scheduler", "linear"),
            logging_steps=10,
            save_steps=cfg.checkpoint.save_steps,
            output_dir=cfg.checkpoint.dirpath,
            report_to="wandb",
            optim="adamw_8bit",    # Unsloth-recommended: saves VRAM vs standard AdamW
            seed=42,
        ),
    )

    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(cfg.checkpoint.dirpath)
    tokenizer.save_pretrained(cfg.checkpoint.dirpath)
    print(f"\nAdapter saved to {cfg.checkpoint.dirpath}")

    # Export to GGUF
    quant = cfg.checkpoint.get("gguf_quantization", "q4_k_m")
    gguf_path = cfg.checkpoint.dirpath + "_gguf"
    print(f"Exporting GGUF ({quant}) → {gguf_path}")
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method=quant)
    print(f"GGUF saved to {gguf_path}")

    wandb.finish()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train_llm.yaml"
    train(config_path)
