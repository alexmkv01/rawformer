"""A from-scratch transformer and neural network library built with NumPy."""

from rawformer.attention import MultiHeadAttention
from rawformer.core import ForwardNotCalledError, Layer, ShapeMismatchError, SimpleLayer
from rawformer.layers import (
    Dropout,
    IdentityLayer,
    LayerNorm,
    LinearLayer,
    PositionalEncoding,
    ReluLayer,
    RMSNorm,
    SigmoidLayer,
    TanhLayer,
    TokenEmbedding,
    he_init,
    xavier_init,
    zeros_init,
)
from rawformer.losses import CrossEntropyLoss, Loss, MSELoss
from rawformer.models import MultiLayerNetwork
from rawformer.preprocessing import Preprocessor
from rawformer.tokenizers import BPETokenizer, WordPieceTokenizer
from rawformer.training import (
    DecoderOnlyModel,
    LMTrainer,
    MLMHead,
    Trainer,
    TrainerHyperparams,
    mask_tokens,
)
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
    "DecoderOnlyModel",
    "Dropout",
    "Encoder",
    "EncoderBlock",
    "ForwardNotCalledError",
    "IdentityLayer",
    "LMTrainer",
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
    "RMSNorm",
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
