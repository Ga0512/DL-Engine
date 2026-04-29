"""
Stream text samples from WebDataset tar shards stored in S3.

Shard format inside each .tar:
    {key}.json  — one training sample per file

Supported JSON schemas:

  Alpaca format (cfg.data.format = "alpaca"):
    {"instruction": "...", "input": "...", "output": "..."}
    "input" is optional.

  Chat format (cfg.data.format = "chat"):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    Formatted using the tokenizer's built-in chat template.

  Raw format (cfg.data.format = "text"):
    {"text": "..."}
    Used as-is — full prompt+response already formatted.
"""

import json
import webdataset as wds
from datasets import IterableDataset
from omegaconf import DictConfig

from src.loaders.webdataset_loader import _expand_to_pipe_urls

_ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}"
    "{input_block}"
    "\n\n### Response:\n{output}"
)


def _format_alpaca(sample: dict) -> str:
    input_block = f"\n\n### Input:\n{sample['input']}" if sample.get("input") else ""
    return _ALPACA_TEMPLATE.format(
        instruction=sample["instruction"],
        input_block=input_block,
        output=sample["output"],
    )


def _format_chat(sample: dict, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def build_text_dataset(cfg: DictConfig, tokenizer=None) -> IterableDataset:
    """
    Returns a HuggingFace IterableDataset that streams from S3 shards.
    Suitable for passing directly to SFTTrainer (dataset_text_field="text").
    """
    urls = _expand_to_pipe_urls(cfg.data.train_shards)
    fmt = cfg.data.get("format", "alpaca")

    if fmt == "chat" and tokenizer is None:
        raise ValueError("format='chat' requires tokenizer to apply the chat template")

    def _generator():
        dataset = (
            wds.WebDataset(urls, shardshuffle=500, nodesplitter=wds.split_by_node)
            .shuffle(2000)
            .decode()
            .to_tuple("json")
        )
        for (raw,) in dataset:
            sample = raw if isinstance(raw, dict) else json.loads(raw)
            if fmt == "alpaca":
                yield {"text": _format_alpaca(sample)}
            elif fmt == "chat":
                yield {"text": _format_chat(sample, tokenizer)}
            else:
                yield {"text": sample["text"]}

    return IterableDataset.from_generator(_generator)
