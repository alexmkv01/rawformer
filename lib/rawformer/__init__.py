"""A from-scratch transformer and neural network library built with NumPy."""

from rawformer.attention import MultiHeadAttention
from rawformer.base import Layer, SimpleLayer
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
from rawformer.tokenizers import BPETokenizer, WordPieceTokenizer
from rawformer.trainer import Trainer, TrainerHyperparams
from rawformer.training import MLMHead, mask_tokens
from rawformer.transformer import (
    Decoder,
    DecoderBlock,
    Encoder,
    EncoderBlock,
    PositionWiseFeedForward,
    Transformer,
    causal_mask,
)

__all__ = [
    "BPETokenizer",
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
    "MLMHead",
    "MSELoss",
    "MultiHeadAttention",
    "MultiLayerNetwork",
    "PositionWiseFeedForward",
    "PositionalEncoding",
    "Preprocessor",
    "ReluLayer",
    "ShapeMismatchError",
    "SigmoidLayer",
    "SimpleLayer",
    "TanhLayer",
    "TokenEmbedding",
    "Trainer",
    "TrainerHyperparams",
    "Transformer",
    "WordPieceTokenizer",
    "causal_mask",
    "he_init",
    "mask_tokens",
    "xavier_init",
    "zeros_init",
]
