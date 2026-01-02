"""TinyStories dataset loading.

This module handles:
- Loading TinyStories from HuggingFace
- Tokenization with GPT-2 tokenizer
- Creating train/val dataloaders
"""
from typing import Any, Iterator
import re

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from loguru import logger
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer

load_dotenv()  # Load HF_TOKEN


class TokenizedDataset(IterableDataset):
    """Streaming tokenized dataset."""
    
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        seq_len: int,
        accelerator=None,
        add_bos: bool = False,
        bos_token_id: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer: list[int] = []
        self.accelerator = accelerator
        self.add_bos = add_bos
        self.bos_token_id = bos_token_id
    
    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        base_split = self.split
        limit = None
        match = re.match(r"^([^\[]+)\[:([^\]]+)\]$", self.split)
        if match:
            base_split = match.group(1)
            slice_str = match.group(2).strip()
            if slice_str.endswith("%"):
                try:
                    pct = float(slice_str[:-1])
                    from datasets import load_dataset_builder
                    builder = load_dataset_builder(self.dataset_name)
                    split_info = builder.info.splits.get(base_split)
                    if split_info is not None and split_info.num_examples is not None:
                        limit = max(1, int(split_info.num_examples * pct / 100))
                    else:
                        logger.warning(
                            f"Cannot resolve split size for {self.dataset_name}:{base_split}; ignoring slice {self.split}"
                        )
                except Exception:
                    logger.warning(f"Failed to parse split slice {self.split}; ignoring slice")
            else:
                try:
                    limit = max(1, int(slice_str))
                except ValueError:
                    logger.warning(f"Failed to parse split slice {self.split}; ignoring slice")

        dataset = load_dataset(self.dataset_name, split=base_split, streaming=True)
        if limit is not None:
            dataset = dataset.take(limit)

        num_shards = 1
        shard_id = 0
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            num_shards *= self.accelerator.num_processes
            shard_id = self.accelerator.process_index
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            num_shards *= worker_info.num_workers
            shard_id = shard_id * worker_info.num_workers + worker_info.id
        if num_shards > 1:
            dataset = dataset.shard(
                num_shards=num_shards,
                index=shard_id,
                contiguous=True,
            )
        
        buffer = self.buffer
        for sample in dataset:
            text = sample["text"]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_bos and self.bos_token_id is not None:
                buffer.append(self.bos_token_id)
            buffer.extend(tokens)
            
            while len(buffer) >= self.seq_len + 1:
                x_tokens = buffer[: self.seq_len]
                y_tokens = buffer[1 : self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                
                yield {
                    "input_ids": torch.tensor(x_tokens, dtype=torch.long),
                    "labels": torch.tensor(y_tokens, dtype=torch.long),
                }
        
        # Keep leftover buffer for next epoch
        self.buffer = buffer


def get_tokenizer(name: str = "gpt2"):
    """Load tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
    logger.info(f"Loaded tokenizer '{name}' (vocab={tokenizer.vocab_size})")
    return tokenizer


def get_dataloaders(cfg, accelerator=None) -> tuple[DataLoader, DataLoader, Any]:
    """Create train and val dataloaders.
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    tokenizer = get_tokenizer(cfg.data.tokenizer_name)
    tokenizer.model_max_length = cfg.model.max_seq_len
    
    add_bos = getattr(cfg.data, "add_bos", False)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    train_dataset = TokenizedDataset(
        cfg.data.dataset_name,
        cfg.data.train_split,
        tokenizer,
        cfg.model.max_seq_len,
        accelerator=accelerator,
        add_bos=add_bos,
        bos_token_id=bos_id,
    )
    val_dataset = TokenizedDataset(
        cfg.data.dataset_name,
        cfg.data.val_split,
        tokenizer,
        cfg.model.max_seq_len,
        accelerator=accelerator,
        add_bos=add_bos,
        bos_token_id=bos_id,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.device_batch_size,
        num_workers=min(cfg.data.num_workers, 2),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.device_batch_size,
        num_workers=min(cfg.data.num_workers, 2),
        pin_memory=True,
        drop_last=True,
    )
    
    return train_loader, val_loader, tokenizer
