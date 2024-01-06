from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset, DatasetDict, Dataset
from tiktoken import get_encoding
from functools import partial
import numpy as np
from tqdm import tqdm
from math import ceil
import os

@dataclass
class DatasetManager:
    dataset: str = "bellthomas/herodotus"
    data_path: Path = Path(os.path.dirname(__file__)).absolute()
    n_proc: int = 16
    trust_remote_code: bool = True
    training_split: str = "train"
    validation_split: Optional[str] = None
    test_set_ratio: float = 0.05
    feature_name: str = "text"
    encoding = get_encoding("gpt2")


    def prepare(self) -> None:
        print(f"Preparing dataset: {self.dataset}")
        dataset_safe_id = self.dataset.replace("/", "-")
        dataset_path = self.data_path / dataset_safe_id
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Load dataset from Hugging Face.
        # For testing: `split="train[:2%]"`
        dataset = load_dataset(self.dataset, num_proc=self.n_proc, trust_remote_code=self.trust_remote_code)
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({ self.training_split: dataset })
        splits = ", ".join(list(dataset.keys()))
        print(f"Native splits: ({splits})")
        dataset = self.validate(dataset)

        # Encode splits.
        encoder = partial(DatasetManager.encode, self)
        dataset = dataset.map(encoder, remove_columns=[self.feature_name], num_proc=self.n_proc, load_from_cache_file=False)

        # Join into single, large blob for each split for downstream consumption.
        for split, dataset in dataset.items():
            length = np.sum(dataset['count'], dtype=np.uint64)
            buffer = np.memmap(dataset_path / split, dtype=np.uint16, mode='w+', shape=(length,))
            total_batches = ceil(length / (10 * 1024 * 1204))  # Approx. 10MB/batch.

            token_idx = 0
            label = f"Writing {dataset_safe_id}/{split}"
            for batch_idx in tqdm(range(total_batches), desc=label):
                batch = dataset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                batch_data = np.concatenate(batch['data'])
                buffer[token_idx : token_idx + len(batch_data)] = batch_data
                token_idx += len(batch_data)
            buffer.flush()

    def validate(self, dataset: DatasetDict) -> DatasetDict:
        #
        if self.encoding.max_token_value >= 2**16:
            print("Encoding can't fit into uint16!")
            exit(3)

        # 
        if self.training_split not in dataset:
            print(f"Training split '{self.training_split}' not found.")
            exit(1)

        if self.validation_split and self.validation_split not in dataset:
            print(f"Validation split '{self.validation_split}' not found.")
            exit(2)

        # If we don't have a declared validation split we need to generate one.
        # Default: minimum of 0.5% or 
        if not self.validation_split:
            _minimum_rows = 10  # Arbitrary for now.
            rows = dataset[self.training_split].num_rows
            if (self.test_set_ratio * rows) < _minimum_rows:
                self.test_set_ratio = rows / _minimum_rows
                print(f"Increasing validation split size to {self.test_set_ratio}.")

            dataset = dataset[self.training_split].train_test_split(test_size=self.test_set_ratio, shuffle=True)
            self.validation_split = "test"
        
        return dataset

    @staticmethod
    def encode(cls, data) -> dict:
        data = cls.encoding.encode_ordinary(data[cls.feature_name])
        data.append(cls.encoding.eot_token)  # Force end-of-text token to be present.
        return {"data": data, "count": len(data)}
