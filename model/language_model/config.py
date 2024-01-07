from dataclasses import dataclass
from typing import Union

@dataclass
class Config:
    layers: int = 8
    attention_heads: int = 8
    embedding_size: int = 512
    block_size: int = 512
    bias: Union[bool, float] = False  # Set to False to disable.
    vocab_size: int = 50304
    dropout: float = 0.0
    device: str = "cpu"
    batch_size: int = 8
    gradient_accumulation_steps: int = 1

    def head_size(self) -> int:
        return self.embedding_size // self.attention_heads
