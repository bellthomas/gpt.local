import argparse
import os
import numpy as np
from pathlib import Path
import tiktoken
import requests
from datasets import load_dataset

from .dataset import DatasetManager

data = DatasetManager("openwebtext")
try:
    data.prepare()
except KeyboardInterrupt:
    exit(0)
exit(0)

collections = {
    "herodotus": "https://gist.githubusercontent.com/bellthomas/9c776e96f58afaa584585060c7f1e8d6/raw/7be7fdf402b4845d748480dadf2d45329b0bb1c7/herodotus_histories.txt",
    "openwebtext": "openwebtext"
}

if __name__ == "__main__":
    # Handle arguments.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--collection', default="herodotus", help='')
    parser.add_argument('--split', help='', type=float, default=0.9)
    args = parser.parse_args()

    if not args.collection:
        print("No collection specified.")
        exit(1)

    path = Path(f"{os.path.dirname(__file__)}/{args.collection}").absolute()
    path.mkdir(parents=True, exist_ok=True)

    # If we don't have a data file, download it.
    if not os.path.isfile(path / "data.txt"):
        if not args.collection in collections:
            exit("Unknown")
        
        if collections[args.collection].startswith("http"):
            print(f"Downloading collection: {args.collection}")
            response = requests.get(collections[args.collection])
            if response.status_code == 200:
                with open(path / "data.txt", "w") as f:
                    f.write(response.text)
            else:
                exit("Failed to download file.")
        else:
            # Loading using huggingface `datasets`.
            print(f"Downloading collection using huggingface: {args.collection}")
            dataset = load_dataset("openwebtext", num_proc=8, trust_remote_code=True)
            split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
            print(split_dataset)
            print("---")
            encoding = tiktoken.get_encoding("gpt2")

            def process(data):
                return {}
            new_dataset = split_dataset.map(process, num_proc=16)
            print(new_dataset)
            exit(0)

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
