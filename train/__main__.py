import argparse
import data as D
import os
from .train import Trainer
from pathlib import Path

# Handle arguments.
parser = argparse.ArgumentParser(description='')
parser.add_argument('--collection', default="herodotus", help='')
args = parser.parse_args()

# TODO: validation, use args.
data_path = Path(f"{os.path.dirname(D.__file__)}/{args.collection}").absolute()
Trainer.train(args, data_path)