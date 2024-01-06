from dataclasses import dataclass
from typing import Any

from .language_model.Config import Config

@dataclass
class Checkpoint:
    config: Config
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    iteration: int
    training_history: float
    loss: float

    def __str__(self) -> str:
        fields: list[str] = [
            f"iterations={self.iteration}",
            f"training={self.training_history:.2f}s",
            f"loss={self.loss:.4f}",
            f"avg={1000 * self.training_history / self.iteration:.2f}ms"
        ]
        return f"Checkpoint({', '.join(fields)})"
