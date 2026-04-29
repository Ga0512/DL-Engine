"""
Merge LoRA adapter into the base model and export to GGUF.

Usage:
    python scripts/export_gguf.py configs/train_llm.yaml \\
        --adapter  ./checkpoints/llm \\
        --output   ./output_gguf \\
        --quant    q4_k_m

Quantization options: q4_k_m (default), q8_0, f16, q2_k, q5_k_m
"""

import argparse
import sys
from omegaconf import OmegaConf


def export(cfg_path: str, adapter_path: str, output_path: str, quant: str) -> None:
    from unsloth import FastLanguageModel

    cfg = OmegaConf.load(cfg_path)

    print(f"Loading base model + adapter from {adapter_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=cfg.model.max_seq_length,
        dtype=None,
        load_in_4bit=False,   # full precision for clean merge before export
    )

    print(f"Exporting to GGUF ({quant}) → {output_path}")
    model.save_pretrained_gguf(output_path, tokenizer, quantization_method=quant)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config",            help="Path to train_llm.yaml")
    parser.add_argument("--adapter",  required=True, help="Path to saved LoRA adapter")
    parser.add_argument("--output",   default="./output_gguf")
    parser.add_argument("--quant",    default="q4_k_m")
    args = parser.parse_args()

    export(args.config, args.adapter, args.output, args.quant)
