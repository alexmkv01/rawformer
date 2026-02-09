"""A from-scratch transformer and neural network library built with NumPy."""

from rawformer.attention import MultiHeadAttention
from rawformer.base import Layer
from rawformer.exceptions import ForwardNotCalledError, ShapeMismatchError
from rawformer.layers import (
    Dropout,
    IdentityLayer,
    LayerNorm,
    LinearLayer,
    PositionalEncoding,
    ReluLayer,
    SigmoidLayer,
    TanhLayer,
    TokenEmbedding,
    he_init,
    xavier_init,
    zeros_init,
)
from rawformer.losses import CrossEntropyLoss, Loss, MSELoss
from rawformer.network import MultiLayerNetwork
from rawformer.preprocessing import Preprocessor
from rawformer.trainer import Trainer, TrainerHyperparams
from rawformer.transformer import (
    Decoder,
    DecoderBlock,
    Encoder,
    EncoderBlock,
    PositionWiseFeedForward,
    Transformer,
)

__all__ = [
    "CrossEntropyLoss",
    "Decoder",
    "DecoderBlock",
    "Dropout",
    "Encoder",
    "EncoderBlock",
    "ForwardNotCalledError",
    "IdentityLayer",
    "Layer",
    "LayerNorm",
    "LinearLayer",
    "Loss",
    "MSELoss",
    "MultiHeadAttention",
    "MultiLayerNetwork",
    "PositionWiseFeedForward",
    "PositionalEncoding",
    "Preprocessor",
    "ReluLayer",
    "ShapeMismatchError",
    "SigmoidLayer",
    "TanhLayer",
    "TokenEmbedding",
    "Trainer",
    "TrainerHyperparams",
    "Transformer",
    "he_init",
    "xavier_init",
    "zeros_init",
]
