from dataclasses import dataclass
from typing import Union
from pathlib import Path

@dataclass
class Config:
    layers: int = 12
    attention_heads: int = 8
    embedding_size: int = 768
    block_size: int = 512
    bias: Union[bool, float] = False  # Set to False to disable.
    vocab_size: int = 50304
    dropout: float = 0.0
    device: str = "cpu"
    path: Path = None

    def head_size(self) -> int:
        return self.embedding_size // self.attention_heads
