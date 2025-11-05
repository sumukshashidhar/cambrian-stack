from __future__ import annotations

import tiktoken

# -----------------------------------------------------------------------------
# Tokenizer (GPT-2 via tiktoken)
# -----------------------------------------------------------------------------
class Tokenizer:
    def __init__(self, name: str="gpt2"):
        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab
        self.eot = self.enc.eot_token

    @property
    def eos_token_id(self) -> int: return self.eot
    def encode(self, text: str) -> list[int]: return self.enc.encode_ordinary(text)
    def decode(self, ids: list[int]) -> str: return self.enc.decode(ids)

def _tokenize_text(args: tuple[str, str, int]) -> list[int]:
    """Helper for parallel tokenization (must be top-level for pickling)"""
    text, enc_name, eos = args
    enc = tiktoken.get_encoding(enc_name)
    toks = enc.encode_ordinary(text)
    toks.append(eos)
    return toks

