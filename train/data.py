import torch
from pathlib import Path
import numpy as np
from .language_model.Config import Config

class DataLoader:
    splits: set[str] = {"training", "validation"}
    config: Config
    data_path: Path

    def __init__(self, config: Config, data_path: Path) -> None:
        self.config = config
        self.data_path = data_path
        print(f"Data: {self.data_path}/" + "{" + ','.join(self.splits) + "}")

    #
    def fetch_batch(self, split: str, samples: int):
        if split in self.splits:
            device = self.config.device
            bs = self.config.block_size
            data = np.memmap(self.data_path / split, dtype=np.uint16, mode='r')
            indices = torch.randint(len(data) - bs, (samples,))

            def _extract_sample(start: int):
                return torch.from_numpy((data[start:start+bs]).astype(np.int64))
            
            batch_sequences = torch.stack([_extract_sample(i) for i in indices]).to(device)
            batch_next = torch.stack([_extract_sample(i + 1) for i in indices]).to(device)
            return batch_sequences, batch_next
        
        exit("Unknown split.")