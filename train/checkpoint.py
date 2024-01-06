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

    def average_step_duration_ms(self) -> float:
        return 1000 * self.training_history / self.iteration

    def csv_row(self) -> str:
        items = (
            self.iteration,
            self.loss,
            self.training_history,
            self.average_step_duration_ms()
        )
        return ",".join([f"{x}" for x in items])


    def __str__(self) -> str:
        fields: list[str] = [
            f"iterations={self.iteration}",
            f"training={self.training_history:.2f}s",
            f"loss={self.loss:.4f}",
            f"avg={self.average_step_duration_ms():.2f}ms"
        ]
        return f"Checkpoint({', '.join(fields)})"
