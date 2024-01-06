# gpt.local

A **work-in-progress**, *from-scratch* implementation of a generative pre-trained transformer (GPT) in vanilla Pytorch. The purpose of this project is to be a personal sandbox + learning environment, looking at both training and inference, with a long-term aim of becoming a self-hosted language assistant running in a homelab environment.

Inspired by OpenAI's [GPT 2](https://github.com/openai/gpt-2), Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), and Hugging Face's [GPT 2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py).

---

#### Getting Started
```bash
git clone git@github.com:bellthomas/gpt.local.git
cd gpt.local

# Download training data.
python -m data
> Downloading collection: bellthomas/herodotus
> ...


# Train a GPT.
python -m train
> Initialised data path: ./gpt.local/data/herodotus
> Step 0: loss 10.9303 (2385.89ms)
> Step 1: loss 10.9261 (1333.66ms)
> Step 2: loss 10.9168 (1496.20ms)
> ...
```

---

#### To implement/explore...

 - [x] Hugging Face dataset integration.
 - [ ] CUDA support
 - [ ] Checkpoint saving
 - [ ] Text generation from trained model
 - [ ] Checkpoint loading
 - [ ] Improve performance to bring in-line with other PyTorch GPT implementations.
 - [ ] Variable-precision.
 - [ ] Flash attention.
 - [ ] Implement in a lower-level language: Rust?
 - [ ] [*Faster LLMs on _low_-_memory_ devices*](https://arxiv.org/pdf/2312.11514.pdf).
 - [ ] ...