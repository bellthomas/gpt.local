import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from .Transformer import Transformer
from .Config import Config
from ..data import DataLoader


class LanguageModel(nn.Module):
    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        # Initialise primary components.
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.positional_embedding = nn.Embedding(config.block_size, config.embedding_size)
        self.transformer_layers = nn.Sequential(*[Transformer(config) for _ in range(config.layers)])
        self.normalise = nn.LayerNorm(config.embedding_size) # Normalisation at the end.
        self.decode = nn.Linear(config.embedding_size, config.vocab_size)

        # Recursively initialise submodules.
        self.apply(self._initialise_module_weights)

    #
    def _initialise_module_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    #
    def forward(self, sample, targets=None) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # `sample` and `targets` are both (B, T) tensors of ints.
        B, T = sample.shape

        # NB. torch.arange(X) -> torch.Tensor([0, 1, 2, 3, ..., X-1])
        _range = torch.arange(T, device=self.config.device) # shape: (T)
        _token = self.token_embedding(sample) # shape: (B, T, C)
        _position = self.positional_embedding(_range) # shape: (T, C)
        logits = self.transformer_layers(_token + _position) # (B, T, C) -> (B, T, C)
        logits = self.normalise(logits) # (B, T, C)
        predictions = self.decode(logits) # shape: (B, T, config.vocab_size)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            # Can only compute loss if we have target comparisons.
            # Reshape `predictions` and `targets` as needed and compute `cross_entropy(...)`.
            B, T, C = predictions.shape
            predictions = predictions.view(B * T, C) # shape: (B * T, C)
            targets = targets.view(B * T) # shape: (B * T)
            loss = F.cross_entropy(predictions, targets) # fn: (N, C, ...) x (C, ...) -> ()

        return predictions, loss

    def generate(self, sample, tokens_to_generate: int):
        # `sample` is (B, T) tensor: the prediction context for each batch.
        for _ in range(tokens_to_generate):
            # Steps:
            #   1. Crop sample's length to be a maximum of the block size.
            #   2. Execute forward pass to generate predictions.
            #   3. Extract the final index (logits representing the final, newly-generated token).
            #   4. Softmax & sample to get successor token.
            #   5. Append generated token to sample to maintain context for future predictions.
            cropped_sample = sample[:, -self.config.block_size:]
            prediction, _ = self(cropped_sample)
            final_indices = prediction[:, -1, :] # shape: (B, C)
            successor_probabilities = F.softmax(final_indices, dim=-1) # shape: (B, C)
            successor = torch.multinomial(successor_probabilities, num_samples=1) # shape: (B, 1)
            sample = torch.cat((sample, successor), dim=1) # shape: (B, T+1)
        return sample

    #
    @torch.no_grad()
    def estimate_loss(self, data: DataLoader, batch_size: int) -> dict[str, torch.Tensor]:
        self.eval()

        def _loss(split: str) -> float:
            seq, next = data.fetch_batch(split, batch_size)
            _, loss = self(seq, next)
            return loss.item()

        def _mean_loss(split: str, samples: int = 1) -> torch.Tensor:
            losses = torch.tensor([_loss(split) for _ in range(samples)])
            return losses.mean()

        result = {
            split: _mean_loss(split, samples=batch_size)
            for split in data.splits
        }
        self.train()  # Reset.
        return result

    #
    def optimizer(self, weight_decay, learning_rate, betas):
        parameters = { x: y for x, y in self.named_parameters() if y.requires_grad }

        # Apply weight decay to non-linear parameters (matrix-multiplications/embeddings).
        return torch.optim.AdamW(
            [
                {'params': [p for p in parameters.values() if p.dim() >= 2], 'weight_decay': weight_decay},
                {'params': [p for p in parameters.values() if p.dim() < 2], 'weight_decay': 0.0}
            ],
            lr=learning_rate,
            betas=betas
        )
