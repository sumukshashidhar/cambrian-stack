"""TinyStories dataset loading.

This module handles:
- Loading TinyStories from HuggingFace
- Tokenization with GPT-2 tokenizer
- Creating train/val dataloaders
"""
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from loguru import logger
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer

load_dotenv()  # Load HF_TOKEN


class TokenizedDataset(IterableDataset):
    """Streaming tokenized dataset."""
    
    def __init__(self, dataset_name: str, split: str, tokenizer, seq_len: int, accelerator=None):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer: list[int] = []
        self.accelerator = accelerator
    
    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            dataset = dataset.shard(
                num_shards=self.accelerator.num_processes,
                index=self.accelerator.process_index,
                contiguous=True,
            )
        
        buffer = self.buffer
        for sample in dataset:
            text = sample["text"]
            tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=self.seq_len + 1)
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
    logger.info(f"Loaded tokenizer '{name}' (vocab={tokenizer.vocab_size})")
    return tokenizer


def get_dataloaders(cfg, accelerator=None) -> tuple[DataLoader, DataLoader, any]:
    """Create train and val dataloaders.
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    tokenizer = get_tokenizer(cfg.data.tokenizer_name)
    tokenizer.model_max_length = cfg.model.max_seq_len
    
    train_dataset = TokenizedDataset(
        cfg.data.dataset_name,
        cfg.data.train_split,
        tokenizer,
        cfg.model.max_seq_len,
        accelerator=accelerator,
    )
    val_dataset = TokenizedDataset(
        cfg.data.dataset_name,
        cfg.data.val_split,
        tokenizer,
        cfg.model.max_seq_len,
        accelerator=accelerator,
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
