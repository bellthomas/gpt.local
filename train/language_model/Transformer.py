import torch.nn as nn
from .FeedForward import FeedForward
from .AttentionEnsemble import AttentionEnsemble
from .Config import Config

class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention_layer = AttentionEnsemble(config)
        self.feed_forward_layer = FeedForward(config)
        self.normalise_1 = nn.LayerNorm(config.embedding_size)
        self.normalise_2 = nn.LayerNorm(config.embedding_size)

    def forward(self, sample):
        step_1 = sample + self.attention_layer(self.normalise_1(sample))
        step_2 = step_1 + self.feed_forward_layer(self.normalise_2(step_1))
        return step_2