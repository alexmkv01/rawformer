# Neural Network Mini-Library

A from-scratch neural network library built with NumPy, plus a fully trained classifier on the Iris dataset — structured as a modern Python monorepo with a reproducible DVC training pipeline.

Originally created as coursework for the Introduction to Machine Learning module at Imperial College London, now refactored with professional ML engineering practices.

## Papers & Theory

Implements from scratch:

- Backpropagation and gradient descent as formalised in *Learning representations by back-propagating errors* (Rumelhart, Hinton & Williams, 1986)
- Xavier/Glorot weight initialisation — *Understanding the difficulty of training deep feedforward neural networks* (Glorot & Bengio, 2010)
- He/Kaiming initialisation for ReLU networks — *Delving Deep into Rectifiers* (He et al., 2015)
- Softmax cross-entropy loss with the log-sum-exp trick for numerical stability — *Training Stochastic Model Recognition Algorithms as Networks can Lead to Maximum Mutual Information Estimation of Parameters* (Bridle, 1989)

## Project Structure

```
.
├── lib/                          # nn-lib: the neural network library
│   ├── pyproject.toml
│   └── nn_lib/
│       ├── base.py               # Abstract Layer base class
│       ├── activations.py        # ReLU, Sigmoid, Tanh, Identity
│       ├── layers.py             # LinearLayer (fully connected)
│       ├── losses.py             # MSE, Cross-Entropy with softmax
│       ├── network.py            # MultiLayerNetwork
│       ├── trainer.py            # Mini-batch SGD trainer
│       ├── preprocessing.py      # Min-max normalization
│       ├── initializers.py       # Xavier, He, zeros
│       └── tests/                # Unit tests with numerical gradient checks
├── train/                        # nn-train: DVC training pipeline
│   ├── pyproject.toml
│   └── nn_train/
│       ├── prepare.py            # Data loading, splitting, preprocessing
│       ├── train.py              # Model construction and training
│       ├── evaluate.py           # Evaluation and metrics output
│       └── tests/                # Pipeline smoke tests
├── data/                         # DVC-tracked data (fetched via dvc pull)
│   └── iris.dat.dvc
├── artifacts/                    # Pipeline outputs (reproduced via dvc repro)
│   └── metrics.json
├── dvc.yaml                      # Pipeline definition
├── params.yaml                   # Hyperparameters
├── pyproject.toml                # uv workspace root
└── .github/workflows/ci.yml     # Lint, type-check, test
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone <repository-url>
cd Neural_Networks_Mini-Library
uv sync --all-packages
```

## Running the Pipeline

The training pipeline is managed by [DVC](https://dvc.org/):

```bash
# Fetch data (requires configured S3 remote)
uv run dvc pull

# Run the full pipeline: prepare -> train -> evaluate
uv run dvc repro

# View metrics
uv run dvc metrics show
```

### Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| `prepare` | `nn_train.prepare` | Load iris.dat, train/val split, fit preprocessor |
| `train` | `nn_train.train` | Build network from `params.yaml`, train with SGD |
| `evaluate` | `nn_train.evaluate` | Compute val loss and accuracy, write `metrics.json` |

### Hyperparameters

All tunable values live in `params.yaml`:

```yaml
prepare:
  test_split: 0.2
  random_seed: 42

train:
  neurons: [16, 3]
  activations: ["relu", "identity"]
  batch_size: 8
  epochs: 1000
  learning_rate: 0.01
  loss_fun: "cross_entropy"
  shuffle: true
```

## Development

```bash
# Lint and format
uv run ruff format lib/ train/
uv run ruff check lib/ train/

# Type checking (strict)
uv run mypy lib/nn_lib/ train/nn_train/

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=nn_lib --cov=nn_train -v
```

## Library Usage

```python
from nn_lib import MultiLayerNetwork, Preprocessor, Trainer, TrainerHyperparams

import numpy as np

# Preprocess
data = np.loadtxt("data/iris.dat")
x, y = data[:, :4], data[:, 4:]
prep = Preprocessor(x)
x_norm = prep.apply(x)

# Build network: 4 -> 16 (relu) -> 3 (identity + softmax in loss)
net = MultiLayerNetwork(input_dim=4, neurons=[16, 3], activations=["relu", "identity"])

# Train
hyperparams: TrainerHyperparams = {
    "batch_size": 8,
    "nb_epoch": 1000,
    "learning_rate": 0.01,
    "loss_fun": "cross_entropy",
    "shuffle_flag": True,
}
trainer = Trainer(network=net, hyperparams=hyperparams)
trainer.train(x_norm, y)
print(f"Loss: {trainer.eval_loss(x_norm, y):.4f}")
```

## DVC Remote Configuration

The S3 remote is configured as a placeholder. To set up your own:

```bash
# Update the remote URL
uv run dvc remote modify s3remote url s3://your-bucket/path

# Push data
uv run dvc push
```
