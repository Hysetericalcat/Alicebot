# AliceGPT — Transformer from Scratch

A GPT-style language model trained on *Alice in Wonderland*, built from first principles without using HuggingFace model classes.

## What's in here

```
├── tokeniser/          # Trained BPE tokenizer (vocab_size=3000)
├── checkpoints/        # Saved model weights per epoch
├── dataset.py          # Sliding window dataset with train/val split
├── model.py            # Full GPT implementation
│   ├── Self_Attention      # Single causal attention head
│   ├── MultiHeadAttention  # n_head parallel heads + projection
│   ├── MLP                 # Feed-forward block (d_model → 4d → d_model)
│   ├── TransformerBlock    # Attention + MLP + LayerNorm + residuals
│   ├── Embedding           # Token + positional embeddings
│   └── Model               # Full GPT (embed → N blocks → LN → LM head)
├── train.py            # Training loop with AdamW + gradient clipping
└── generate.py         # Inference with greedy / top-k / top-p sampling
```

## Architecture

| Hyperparameter | Value |
|---|---|
| Layers | 4 |
| d_model | 256 |
| Attention heads | 4 |
| d_k (per head) | 64 |
| MLP expansion | 4× (256 → 1024 → 256) |
| Block size | 128 tokens |
| Vocab size | 3000 (BPE) |

## Quickstart

**Install dependencies:**
```bash
pip install torch tokenizers
```

**Train:**
```python
from train import Train

with open("alice_in_wonderland.txt", "r") as f:
    content = f.read()

train = Train(
    content=content,
    vocab_size=3000,
    block_size=128,
    batch_size=32,
    n_head=4,
    d_model=256,
    n_layers=4,
    train_size=0.9
)
train.run(n_epochs=10)
```

**Generate:**
```python
from generate import Generate

g = Generate()
g.generate("What is the game", max_new_tokens=100)
```

**Sample output:**
> What is the game ,' said the Queen merely remarking that a rat s and the cri come once crowded together. `I don't know much,' said the Hatter. He was looking at Alice with great curiosity...

## How it works

**Tokenizer** — BPE trained from scratch on the corpus using the `tokenizers` library. Whitespace pre-tokenizer, 3000 merge operations.

**Dataset** — Encodes the full text into a flat token array, then slides a window of `block_size=128` to produce `(x, y)` pairs where `y = x` shifted one position right. 90/10 train/val split.

**Attention** — Each head computes `softmax(QKᵀ / √d_k) · V` with a causal upper-triangular mask so position `i` cannot attend to `j > i`. Four heads run in parallel, outputs concatenated and projected back to `d_model`.

**Training** — AdamW optimizer, cross entropy loss on next-token prediction, gradient clipping at 1.0. Runs on MPS (Apple Silicon) or CUDA.

## Notes

- Dataset is ~150k characters (~40k tokens after BPE). Model overfits by epoch 3-5, which is expected at this scale.
- Initial loss ≈ `ln(3000) ≈ 8.0` (random baseline). Loss drops to ~0.22 by epoch 3.
- KV cache not implemented — each generation step runs a full forward pass.
