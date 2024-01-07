import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import Config

class Head(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        head_size = config.head_size()
        self.key = nn.Linear(config.embedding_size, head_size, bias=False)
        self.query = nn.Linear(config.embedding_size, head_size, bias=False)
        self.value = nn.Linear(config.embedding_size, head_size, bias=False)
        self.register_buffer('trilangular_mask', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # inputs: (B, T, C), outputs: (B, T, head_size)
        _, T, _ = x.shape
        key = self.key(x) # shape: (B, T, head_size)
        query = self.query(x) # shape: (B, T, head_size)
        value = self.value(x) # shape: (B, T, head_size)

        # Calculate affinities (self-attention).
        head_size = key.shape[-1]
        key = key.transpose(-2,-1) # (B, T, head_size) -> (B, head_size, T)
        weights = query @ key * head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        cropped_triangular_mask = self.trilangular_mask[:T, :T] # shape: (T, T)
        weights = weights.masked_fill(cropped_triangular_mask == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)

        # Weighted-aggregation.
        return weights @ value # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
