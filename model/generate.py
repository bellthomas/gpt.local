from pathlib import Path
import torch
import tiktoken

from .language_model import LanguageModel
from . import Checkpoint, Trainer

class Generator:
    model: LanguageModel
    device: str
    temperature: float = 0.8
    encoding = tiktoken.get_encoding("gpt2")

    def __init__(self, checkpoint_path: Path, device: str) -> None:
        self.device = device
        checkpoint: Checkpoint = torch.load(checkpoint_path)
        model = Trainer.load_checkpoint(checkpoint, Path(), Path()).model
        # model = torch.compile(model)
        # model.to(self.device)
        self.model = model

    def encode(self, data):
        return self.encoding.encode(data, allowed_special={"<|endoftext|>"})

    def decode(self, data):
        return self.encoding.decode(data)


    def run(self, prompt: str = "The colour of the sky is blue but") -> None:
        try:
            buffer = (torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device)[None, ...])

            print("Generating:\n")
            print(prompt, end="", flush=True)
            with torch.no_grad():
                while True:
                    buffer = self.model.generate(buffer, 1, temperature=self.temperature)
                    generated = self.decode(buffer[0].tolist()[-1:])
                    if generated == "<|endoftext|>":
                        break
                    print(generated, end="", flush=True)

        except KeyboardInterrupt:
            print("\nAborting.")
            exit(0)
