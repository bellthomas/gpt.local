import torch.nn as nn
from .config import Config

class FeedForward(nn.Module):
    def __init__(self, config: Config, fanout: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.embedding_size, fanout * config.embedding_size),
            nn.ReLU(),
            nn.Linear(fanout * config.embedding_size, config.embedding_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.network(x)
