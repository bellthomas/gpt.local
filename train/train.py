from pathlib import Path
import math
import itertools
import torch
import time

from . import Checkpoint, DataLoader
from .language_model import LanguageModel, Config

#
class Trainer:
    data_path: Path
    experiment_path: Path
    config: Config

    validation_cadence: int = 50 # steps
    checkpoint_filename: str = "checkpoint"
    checkpoint_path: Path
    log_path: Path

    model: LanguageModel
    optimizer: torch.optim.AdamW
    starting_iteration: int = 0
    training_history: float = 0  # seconds the model has previously been trained for

    def __init__(self, config: Config, experiment_path: Path, data_path: Path) -> None:
        self.config = config
        self.data_path = data_path
        self.experiment_path = experiment_path
        self.model = LanguageModel(self.config).to(self.config.device)
        self.optimizer = self.model.optimizer(1e-1, 6e-4, (0.9, 0.95)) # (weight_decay, learning_rate, (beta1, beta2))
        self.checkpoint_path = self.experiment_path / self.checkpoint_filename
        self.log_path = self.experiment_path / "training.log"

    @staticmethod
    def load(experiment_path: Path, data_path: Path) -> 'Trainer':
        checkpoint_path = experiment_path / "checkpoint"
        if checkpoint_path.exists() and checkpoint_path.is_file():
            checkpoint: Checkpoint = torch.load(checkpoint_path)
            print(checkpoint)
            return Trainer.load_checkpoint(checkpoint, experiment_path, data_path)

        print("Experiment doesn't have any existing checkpoints, can't load.")
        exit(1)

    ##############
        
    def create_checkpoint(self, iteration: int, loss: float) -> Checkpoint:
        return Checkpoint(
            config=self.config,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            iteration=iteration,
            training_history=self.training_history,
            loss=loss,
        )
    
    @staticmethod
    def load_checkpoint(checkpoint: Checkpoint, experiment_path: Path, data_path: Path) -> 'Trainer':
        trainer = Trainer(checkpoint.config, experiment_path, data_path)
        trainer.model.load_state_dict(checkpoint.model_state)
        trainer.model.to(checkpoint.config.device)

        # Recreate optimizer.
        trainer.optimizer = trainer.model.optimizer(1e-1, 6e-4, (0.9, 0.95)) # (weight_decay, learning_rate, (beta1, beta2))
        trainer.optimizer.load_state_dict(checkpoint.optimizer_state)
        trainer.starting_iteration = checkpoint.iteration
        trainer.training_history = checkpoint.training_history
        return trainer


    #
    def execute(self):
        data = DataLoader(self.config, self.data_path)
        print(f"Training... (parameters: {self.model.parameters_count() / 1e6:.2f}M, device: {self.config.device})")

        batch_size = self.config.batch_size
        model = self.model
        optimizer = self.optimizer
        flops_per_iteration = model.flops_per_iteration(batch_size * self.config.gradient_accumulation_steps)

        seq, suc = data.fetch_batch("training", batch_size)
        t0 = time.perf_counter()
        for i in itertools.count(self.starting_iteration):
            # Dynamiclly compute learning rate to use this iteration.
            lr = Trainer.learning_rate(i, 6e-4)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Perform validation at `validation_cadence`.
            if i > self.starting_iteration and i % Trainer.validation_cadence == 0:
                loss_estimate = model.estimate_loss(data, batch_size)
                checkpoint = self.create_checkpoint(i, loss_estimate['validation'].item())
                torch.save(checkpoint, self.experiment_path / "checkpoint")
                with open(self.log_path, "a") as log:
                    log.write(checkpoint.csv_row() + "\n")
                print(checkpoint)
                t0 = time.perf_counter()  # reset

            # Forward pass.
            for _ in range(self.config.gradient_accumulation_steps):
                _, loss = model(seq, suc)
                loss /= self.config.gradient_accumulation_steps
                seq, suc = data.fetch_batch("training", batch_size)
                loss.backward()

            # Clip to remediate exploding gradients + backwards pass.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Stats.
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1
            self.training_history += dt
            lossf = loss.item() * self.config.gradient_accumulation_steps
            flops = flops_per_iteration / dt
            print(f"    ({i}) loss {lossf:.4f} ({dt*1000:.2f}ms, ~{flops/1e12:.2f} tflops)")

    #
    @staticmethod
    def learning_rate(iteration, initial_learning_rate):
        # Cosine-with-warmup implementation.
        warmup_threshold = 2000
        decay_threshold = 600000 # ~ maximum expected iterations
        minimum_learning_rate = initial_learning_rate / 10  # 6e-5

        # Warmup phase.
        if iteration < warmup_threshold:
            return initial_learning_rate * iteration / warmup_threshold

        # Post-decay phase.
        if iteration > decay_threshold:
            return minimum_learning_rate

        # Cosine decay phase.
        decay_ratio = (iteration - warmup_threshold) / (decay_threshold - warmup_threshold)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # range: 0...1
        return minimum_learning_rate + coeff * (initial_learning_rate - minimum_learning_rate)
