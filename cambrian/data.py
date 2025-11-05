from __future__ import annotations

import os, random
from concurrent.futures import ProcessPoolExecutor

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from cambrian.tokenizer import Tokenizer, _tokenize_text

class PackedLMText(Dataset):
    def __init__(self, texts: list[str], tok: Tokenizer, seq_len: int):
        self.seq_len = seq_len
        self.epoch = 0
        eos = tok.eos_token_id
        n_cpus = os.cpu_count() or 1
        logger.info(f"Tokenizing {len(texts)} texts using {n_cpus} CPUs...")
        chunk_base = max(1, n_cpus * 4)
        chunk_size = max(1, len(texts) // chunk_base) if len(texts) else 1
        
        toks: list[int] = []
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            args = [(t, "gpt2", eos) for t in texts]
            for chunk in tqdm(executor.map(_tokenize_text, args, chunksize=chunk_size),
                             total=len(texts), desc="tokenizing", dynamic_ncols=True):
                toks.extend(chunk)
        
        self.tokens = torch.tensor(toks, dtype=torch.long)
        logger.info(f"Total tokens: {len(self.tokens):,}  (seq_len={seq_len})")

    def set_epoch(self, e: int): self.epoch = e
    def __len__(self): return (len(self.tokens) - 1) // self.seq_len
    
    def __getitem__(self, idx: int):
        L = self.seq_len
        rng = random.Random((self.epoch << 32) ^ idx)
        i = rng.randint(0, len(self.tokens) - L - 2)
        chunk = self.tokens[i : i + L + 1]
        return chunk[:-1], chunk[1:]

class PackedEval(Dataset):
    def __init__(self, texts: list[str], tok: Tokenizer, seq_len: int):
        self.seq_len = seq_len
        eos = tok.eos_token_id
        n_cpus = os.cpu_count() or 1
        logger.info(f"Tokenizing {len(texts)} validation texts using {n_cpus} CPUs...")
        chunk_base = max(1, n_cpus * 4)
        chunk_size = max(1, len(texts) // chunk_base) if len(texts) else 1
        
        toks: list[int] = []
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            args = [(t, "gpt2", eos) for t in texts]
            for chunk in tqdm(executor.map(_tokenize_text, args, chunksize=chunk_size),
                             total=len(texts), desc="tokenizing val", dynamic_ncols=True):
                toks.extend(chunk)
        
        self.eos = eos
        self.tokens = torch.tensor(toks, dtype=torch.long)
        if len(self.tokens) < self.seq_len + 1:
            pad = torch.full((self.seq_len + 1 - len(self.tokens),), eos, dtype=torch.long)
            self.tokens = torch.cat((self.tokens, pad))
        self.starts = list(range(0, len(self.tokens) - seq_len - 1, seq_len + 1))
        if not self.starts:
            self.starts = [0]
        logger.info(f"Val tokens: {len(self.tokens):,}  ({len(self.starts)} windows)")
    
    def __len__(self): return len(self.starts)
    
    def __getitem__(self, idx: int):
        s = self.starts[idx]
        chunk = self.tokens[s : s + self.seq_len + 1]
        if chunk.size(0) < self.seq_len + 1:
            pad = torch.full((self.seq_len + 1 - chunk.size(0),), self.eos, dtype=self.tokens.dtype)
            chunk = torch.cat((chunk, pad))
        return chunk[:-1], chunk[1:]

def load_splits(dataset_name: str) -> tuple[list[str], list[str]]:
    train = load_dataset(dataset_name, split="train")
    val = load_dataset(dataset_name, split="validation")
    return list(train["text"]), list(val["text"])

def make_loader(texts: list[str], tok: Tokenizer, seq_len: int, batch_size: int, use_ddp: bool, is_train: bool=True) -> DataLoader:
    ds = PackedLMText(texts, tok, seq_len) if is_train else PackedEval(texts, tok, seq_len)
    nw = min(os.cpu_count(), 8)
    sampler = DistributedSampler(ds, shuffle=is_train) if use_ddp else None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(sampler is None and is_train),
        sampler=sampler, num_workers=nw, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
