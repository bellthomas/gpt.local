from pathlib import Path
import torch
import time
import math
import itertools

from .language_model.LanguageModel import LanguageModel
from .language_model.Config import Config
from .data import DataLoader

#
class Trainer:
    validation_cadence: int = 50 # steps

    #
    @staticmethod
    def train(args, data_path: Path):
        device = "mps"
        config = Config(device=device, path=data_path)
        data = DataLoader(config)

        model = LanguageModel(config).to(device)
        optimizer = model.optimizer(1e-1, 6e-4, (0.9, 0.95)) # (weight_decay, learning_rate, (beta1, beta2))
        batch_size = 8

        seq, suc = data.fetch_batch("training", batch_size)
        t0 = time.time()
        for i in itertools.count():
            # Dynamiclly compute learning rate to use this iteration.
            lr = Trainer.learning_rate(i, 6e-4)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Perform validation at `validation_cadence`.
            if i > 0 and i % Trainer.validation_cadence == 0:
                loss_estimate = model.estimate_loss(data, 1)
                checkpoint = {
                    "config": config,
                    "iteration": i,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, data_path / "checkpoint")
                print(f"(Saving checkpoint, validation loss: {loss_estimate['validation']:.4f})")
                t0 = time.time()  # reset

            # Forward pass.
            accumulation_steps: int = 1
            for _ in range(accumulation_steps):
                _, loss = model(seq, suc)
                loss /= accumulation_steps
                seq, suc = data.fetch_batch("training", batch_size)
                loss.backward()

            # Clip to remediate exploding gradients + backwards pass.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Stats.
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            lossf = loss.item() * accumulation_steps
            print(f"Step {i}: loss {lossf:.4f} ({dt*1000:.2f}ms)")

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

