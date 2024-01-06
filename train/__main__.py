import argparse
import os
from pathlib import Path

import data as D
from . import Trainer

if __name__ == "__main__":
    # Handle arguments.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--collection', default="bellthomas/herodotus", help='')
    args = parser.parse_args()

    # TODO: validation, use args.
    collection_safe_id = args.collection.replace("/", "-")
    data_path = Path(f"{os.path.dirname(D.__file__)}/{collection_safe_id}").absolute()

    try:
        Trainer.train(args, data_path)
    except KeyboardInterrupt:
        print("Aborting.")
        exit(0)