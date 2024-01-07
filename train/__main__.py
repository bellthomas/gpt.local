import argparse
import os
from pathlib import Path
from time import time
import torch

import data as D
from . import Trainer, Config, Checkpoint

if __name__ == "__main__":
    # Handle arguments.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--collection', default="bellthomas/herodotus", help='')
    parser.add_argument('--experiment', default=None, help='')
    parser.add_argument('--device', default="mps", help='')
    parser.add_argument('--status', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # TODO: validation, use args.
    collection_safe_id = args.collection.replace("/", "-")
    data_path = Path(f"{os.path.dirname(D.__file__)}/{collection_safe_id}").absolute()
    cfg = Config(device=args.device)

    # Prepare/load experiment.
    experiments_root = Path(os.path.dirname(__file__)) / "experiments"
    experiment_id = args.experiment if args.experiment else f"{int(time())}_experiment"
    experiment_path = experiments_root / experiment_id
    checkpoint_path = experiment_path / "checkpoint"
    try:
        # Perform status check if requested.
        if args.status:
            if experiment_path.exists():
                if checkpoint_path.is_file() and checkpoint_path.exists():
                    checkpoint: Checkpoint = torch.load(checkpoint_path)
                    print(checkpoint)
                else:
                    print("No checkpoint found.")
            else:
                print("Experiment doesn't exist.")
            exit(0)

        # Create Trainer instance.
        if not experiment_path.exists() or not checkpoint_path.exists():
            print(f"*Experiment: {experiment_id}")
            experiment_path.mkdir(parents=True, exist_ok=True)
            t = Trainer(cfg, experiment_path, data_path)
        else:
            print(f"Experiment: {experiment_id}" )
            t = Trainer.load(experiment_path, data_path)
        t.execute()
    except KeyboardInterrupt:
        print("\nAborting.")
        exit(0)