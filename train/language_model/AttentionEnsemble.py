import torch
import torch.nn as nn
from .Head import Head
from .Config import Config

class AttentionEnsemble(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.heads = nn.ModuleList(Head(config) for _ in range(config.attention_heads))
        self.projection = nn.Linear(config.attention_heads * config.head_size(), config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        head_ensemble = torch.cat([h(x) for h in self.heads], dim=-1)
        projection = self.projection(head_ensemble)
        return self.dropout(projection)
