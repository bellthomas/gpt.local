import argparse
import os
import numpy as np
from pathlib import Path
import tiktoken

collections = {
    "herodotus": "https://gist.githubusercontent.com/bellthomas/9c776e96f58afaa584585060c7f1e8d6/raw/7be7fdf402b4845d748480dadf2d45329b0bb1c7/herodotus_histories.txt"
}

if __name__ == "__main__":
    # Handle arguments.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--collection', help='')
    parser.add_argument('--split', help='', type=float, default=0.9)
    args = parser.parse_args()

    if not args.collection:
        print("No collection specified.")
        exit(1)

    path = Path(f"{os.path.dirname(__file__)}/{args.collection}").absolute()
    # path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.collection}...")

    # Load and split.
    data = ""
    with open(path / "data.txt", "r") as f:
        data = f.read()
    size = len(data)
    divide = int(args.split * size)
    training_data = data[0:divide]
    validation_data = data[divide:]

    # Tokenize.
    encoding = tiktoken.get_encoding("gpt2")
    training_identifiers = encoding.encode_ordinary(training_data)
    validation_identifiers = encoding.encode_ordinary(validation_data)
    print(f"Training: {len(training_identifiers):,} tokens")
    print(f"Validation: {len(validation_identifiers):,} tokens")

    # Write.
    np.array(training_identifiers, dtype=np.uint16).tofile(path / "training")
    np.array(validation_identifiers, dtype=np.uint16).tofile(path / "validation")
