# Rawformer

A from-scratch transformer and neural network library built entirely with NumPy. Implements the full encoder-decoder transformer architecture from "Attention Is All You Need" with complete forward and backward passes, trained and verified against PyTorch.

Originally bootstrapped from coursework for the Introduction to Machine Learning module at Imperial College London, now extended into a complete transformer implementation with professional ML engineering practices.

## Papers & Theory

Implements from scratch with full forward and backward passes:

**Transformer architecture**
- The complete encoder-decoder transformer — *Attention Is All You Need* (Vaswani et al., 2017)
  - Scaled dot-product and multi-head attention
  - Sinusoidal positional encoding
  - Position-wise feed-forward networks
  - Post-norm residual connections (residual + dropout, then layer norm)
  - Causal masking for autoregressive decoding
- Layer normalization — *Layer Normalization* (Ba, Kiros & Hinton, 2016)
- Dropout regularization with inverted scaling — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

**Tokenization**
- Byte Pair Encoding — *Neural Machine Translation of Rare Words with Subword Units* (Sennrich, Haddow & Birch, 2016)
- WordPiece tokenization — *Japanese and Korean Voice Search* (Schuster & Nakajima, 2012)

**Language modelling**
- Masked language modelling (MLM) — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (Devlin et al., 2018)
  - WordPiece tokenization, 80/10/10 masking strategy, MLM prediction head
- Causal language modelling (CLM) with decoder-only architecture — *Improving Language Understanding by Generative Pre-Training* (Radford et al., 2018)
  - BPE tokenization, autoregressive next-token prediction, decoder-only transformer
  - DVC pipeline: pretrain with CLM, then supervised fine-tuning (SFT)

**Feedforward foundations**
- Backpropagation and gradient descent — *Learning representations by back-propagating errors* (Rumelhart, Hinton & Williams, 1986)
- Xavier/Glorot weight initialisation — *Understanding the difficulty of training deep feedforward neural networks* (Glorot & Bengio, 2010)
- He/Kaiming initialisation for ReLU networks — *Delving Deep into Rectifiers* (He et al., 2015)
- Softmax cross-entropy loss with the log-sum-exp trick for numerical stability — *Training Stochastic Model Recognition Algorithms as Networks can Lead to Maximum Mutual Information Estimation of Parameters* (Bridle, 1989)

## Key Design Decisions

- **NumPy only** — no autograd, no frameworks; every gradient is derived and implemented by hand
- **PyTorch cross-verified** — forward passes for attention, multi-head attention, layer norm, embeddings, and the full transformer are tested against PyTorch equivalents
- **Numerical gradient checks** — backward passes are verified via finite-difference approximation for every differentiable layer
- **Strict typing** — mypy strict mode with `disallow_any_explicit`, no `Any` anywhere in the codebase
- **Two-tier layer ABC** — `Layer` as a minimal base for all components, `SimpleLayer` for the standard single-tensor forward/backward contract

## Project Structure

```
.
├── lib/rawformer/                # The library
│   ├── base.py                   # Layer / SimpleLayer ABCs
│   ├── layers/                   # Linear, activations, norm, embeddings, dropout
│   ├── attention/                # Scaled dot-product and multi-head attention
│   ├── transformer/              # Encoder, decoder, full transformer, FFN
│   ├── tokenizers/               # BPE and WordPiece tokenizers
│   │   ├── bpe.py                # Byte Pair Encoding (Sennrich et al., 2016)
│   │   └── wordpiece.py          # WordPiece (Schuster & Nakajima, 2012)
│   ├── training/                 # Language model training utilities
│   │   ├── mlm.py                # Masked language modelling (BERT-style)
│   │   ├── clm.py                # Decoder-only model for causal LM (GPT-style)
│   │   └── lm_trainer.py         # Mini-batch CLM trainer
│   ├── network.py                # Multi-layer feedforward network
│   ├── losses.py                 # MSE, cross-entropy
│   ├── trainer.py                # Mini-batch SGD trainer
│   └── tests/                    # 182 tests including PyTorch cross-verification
├── train/                        # DVC training pipeline
│   └── rawformer_train/          # tokenize -> pretrain -> sft
├── data/                         # Training data (DVC-tracked)
│   ├── pretrain/                 # Text corpus for pretraining
│   └── sft/                      # Instruction-response pairs for SFT
├── dvc.yaml                      # Pipeline definition
├── params.yaml                   # Model and training hyperparameters
├── pyproject.toml                # uv workspace root, ruff + mypy config
└── scripts/lint-and-test.sh      # ruff format + check + mypy + pytest
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone https://github.com/alexmkv01/rawformer.git
cd rawformer
uv sync --all-packages
```

## Development

```bash
# Full lint + type-check + test suite
uv run scripts/lint-and-test.sh

# Or individually:
uv run ruff format
uv run ruff check
uv run mypy .
uv run pytest -v
```

## Library Usage

### Transformer

```python
import numpy as np
from rawformer import Transformer

rng = np.random.default_rng(42)
model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=64,
    n_heads=4,
    n_encoder_layers=2,
    n_decoder_layers=2,
    d_ff=256,
    max_len=128,
    rng=rng,
    dropout_rate=0.1,
)

src = np.array([[1, 5, 10, 3]], dtype=np.intp)
tgt = np.array([[1, 3]], dtype=np.intp)

model.train()
logits = model.forward(src, tgt)    # (1, 2, 1000)
model.backward(np.ones_like(logits))
model.update_params(learning_rate=1e-4)

model.eval()
logits = model.forward(src, tgt)    # dropout disabled
```

### Decoder-Only Language Model

```python
import numpy as np
from rawformer import DecoderOnlyModel, LMTrainer, BPETokenizer

# Train a BPE tokenizer
tokenizer = BPETokenizer()
tokenizer.train(["the cat sat on the mat", "the dog ran in the park"], vocab_size=100)

# Build a tiny decoder-only model
rng = np.random.default_rng(42)
model = DecoderOnlyModel(
    vocab_size=tokenizer.vocab_size,
    d_model=64, n_heads=4, n_layers=2,
    d_ff=256, max_len=128, rng=rng,
)

# Tokenize and train
ids = np.array([tokenizer.encode("the cat sat on the mat")], dtype=np.intp)
trainer = LMTrainer(model=model, learning_rate=0.001, batch_size=1, pad_token_id=0)
loss = trainer.train_epoch(ids, rng)
```

### Feedforward Network

```python
import numpy as np
from rawformer import MultiLayerNetwork, Preprocessor, Trainer, TrainerHyperparams

data = np.loadtxt("data/iris.dat")
x, y = data[:, :4], data[:, 4:]
prep = Preprocessor(x)
x_norm = prep.apply(x)

net = MultiLayerNetwork(input_dim=4, neurons=[16, 3], activations=["relu", "identity"])

hyperparams: TrainerHyperparams = {
    "batch_size": 8,
    "epochs": 1000,
    "learning_rate": 0.01,
    "loss": "cross_entropy",
    "shuffle": True,
}
trainer = Trainer(network=net, hyperparams=hyperparams)
trainer.train(x_norm, y)
print(f"Loss: {trainer.eval_loss(x_norm, y):.4f}")
```

## DVC Pipeline

The training pipeline trains a tiny decoder-only language model:

```bash
uv run dvc repro       # tokenize -> pretrain -> sft
uv run dvc metrics show
```

**Stages:**
1. **tokenize** — Train a BPE tokenizer on the text corpus, encode into token sequences
2. **pretrain** — Train a decoder-only model with causal language modelling (next-token prediction)
3. **sft** — Supervised fine-tuning on instruction-response pairs
