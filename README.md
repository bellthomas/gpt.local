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
python -m train --collection "openwebtext" --experiment "openwebtext-1" --device cpu
> *Experiment: openwebtext-1
> Data: ./gpt.local/data/openwebtext/{validation,training}
> Training... (parameters: 124.11M, device: cpu)
>     (0) loss 10.9385 (9715.26ms, ~0.51 tflops)
>     ...
```

---

#### To implement/explore...

 - [x] Hugging Face dataset integration.
 - [x] Checkpoint saving
 - [x] Checkpoint loading
 - [ ] Text generation from trained model
 - [ ] CUDA support
 - [ ] Improve performance to bring in-line with other PyTorch GPT implementations.
 - [ ] Variable-precision.
 - [ ] Flash attention.
 - [ ] Implement in a lower-level language: Rust?
 - [ ] [*Faster LLMs on _low_-_memory_ devices*](https://arxiv.org/pdf/2312.11514.pdf).
 - [ ] ...