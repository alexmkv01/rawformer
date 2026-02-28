"""Low-Rank Adaptation (LoRA) wrapper for linear layers.

Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
https://arxiv.org/abs/2106.09685

Freezes the base weight matrix W and adds a trainable low-rank
decomposition: y = xW + b + (x @ A @ B) * (alpha / rank).
Only A and B are updated during training, dramatically reducing
the number of trainable parameters.
"""

from typing import TypedDict

import numpy as np
import numpy.typing as npt

from rawformer.core.base import SimpleLayer
from rawformer.core.exceptions import ForwardNotCalledError, ShapeMismatchError
from rawformer.layers.linear import LinearLayer


class _LoRACache(TypedDict):
    input: npt.NDArray[np.float64]


class LoRALinear(SimpleLayer):
    """LoRA wrapper around a frozen LinearLayer.

    Adds a low-rank path ``(x @ A @ B) * (alpha / rank)`` to the
    base linear layer output.  The base layer's weights and biases
    are frozen — only the low-rank matrices A and B are trained.

    Following the paper, B is zero-initialized so the LoRA
    contribution starts at exactly zero, and A uses Kaiming
    initialization (N(0, 1/sqrt(n_in))).

    Supports 2D (batch, features) and 3D (batch, seq_len, features)
    inputs, matching :class:`LinearLayer`.

    Args:
        base: Pre-existing linear layer whose weights are frozen.
        rank: Rank of the low-rank decomposition (r in the paper).
        alpha: Scaling factor (alpha in the paper).
        rng: Random number generator for reproducible initialization.
    """

    def __init__(
        self,
        base: LinearLayer,
        rank: int,
        alpha: float,
        rng: np.random.Generator,
    ) -> None:
        self._base = base
        self._rank = rank
        self._alpha = alpha
        self._scale = alpha / rank

        n_in = base.n_in
        n_out = base.n_out

        # A: Kaiming init  N(0, 1/sqrt(n_in))  — paper convention
        self._lora_a: npt.NDArray[np.float64] = rng.normal(
            loc=0.0, scale=1.0 / np.sqrt(n_in), size=(n_in, rank)
        )
        # B: zero-init so LoRA output starts at exactly zero
        self._lora_b: npt.NDArray[np.float64] = np.zeros((rank, n_out))

        self._cache: _LoRACache | None = None
        self._grad_a: npt.NDArray[np.float64] | None = None
        self._grad_b: npt.NDArray[np.float64] | None = None

    # -- read-only properties ------------------------------------------------

    @property
    def base(self) -> LinearLayer:
        return self._base

    @property
    def lora_a(self) -> npt.NDArray[np.float64]:
        return self._lora_a

    @property
    def lora_b(self) -> npt.NDArray[np.float64]:
        return self._lora_b

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def scale(self) -> float:
        return self._scale

    # -- forward / backward / update -----------------------------------------

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute y = xW + b + (x @ A @ B) * scale.

        Args:
            x: Input array, 2D or 3D with last dim equal to n_in.

        Returns:
            Output array with the same leading dimensions and last dim
            equal to n_out.
        """
        if x.ndim not in (2, 3):
            raise ShapeMismatchError(f"Expected 2D or 3D input, got {x.ndim}D")
        if x.shape[-1] != self._base.n_in:
            raise ShapeMismatchError(f"Last dim {x.shape[-1]} != n_in {self._base.n_in}")

        self._cache = _LoRACache(input=x)

        # Bypass base.forward() to avoid populating its _input_cache and
        # _grad_weights/_grad_biases — the base layer is frozen so we
        # must not accumulate state that implies it will be updated.
        base_out: npt.NDArray[np.float64] = (
            np.einsum("...i,ij->...j", x, self._base.weights) + self._base.biases
        )

        # LoRA path: (x @ A @ B) * scale
        lora_out: npt.NDArray[np.float64] = (
            np.einsum(
                "...i,ij->...j",
                np.einsum("...i,ij->...j", x, self._lora_a),
                self._lora_b,
            )
            * self._scale
        )

        result: npt.NDArray[np.float64] = base_out + lora_out
        return result

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute gradients for A, B, and the input.

        The base weights W are frozen — no gradients are accumulated for them.

        Gradient derivations (s = alpha / rank):
            dL/dx = grad_z @ W.T + (grad_z @ B.T @ A.T) * s
            dL/dA = x.T @ (grad_z @ B.T) * s        (batch dims flattened)
            dL/dB = (x @ A).T @ grad_z * s           (batch dims flattened)

        Args:
            grad_z: Upstream gradient, same shape as the forward output.

        Returns:
            Gradient with respect to the input, same shape as the
            original input to forward.
        """
        if self._cache is None:
            raise ForwardNotCalledError("LoRALinear")

        x = self._cache["input"]
        s = self._scale

        # -- input gradient through base path: grad_z @ W.T --
        grad_input_base: npt.NDArray[np.float64] = np.einsum(
            "...j,ij->...i", grad_z, self._base.weights
        )

        # -- input gradient through LoRA path: (grad_z @ B.T @ A.T) * s --
        grad_input_lora: npt.NDArray[np.float64] = (
            np.einsum(
                "...j,ij->...i",
                np.einsum("...j,ij->...i", grad_z, self._lora_b),
                self._lora_a,
            )
            * s
        )

        # -- parameter gradients (flatten batch dims) --
        x_flat = x.reshape(-1, self._base.n_in)
        grad_flat = grad_z.reshape(-1, self._base.n_out)

        # dL/dB = (x @ A).T @ grad_z * s
        x_a = x_flat @ self._lora_a  # (N, rank)
        self._grad_b = (x_a.T @ grad_flat) * s

        # dL/dA = x.T @ (grad_z @ B.T) * s
        grad_bt = grad_flat @ self._lora_b.T  # (N, rank)
        self._grad_a = (x_flat.T @ grad_bt) * s

        result: npt.NDArray[np.float64] = grad_input_base + grad_input_lora
        return result

    def update_params(self, learning_rate: float) -> None:
        """Update only the LoRA matrices A and B; base weights stay frozen."""
        if self._grad_a is None or self._grad_b is None:
            raise ForwardNotCalledError("LoRALinear")
        self._lora_a -= learning_rate * self._grad_a
        self._lora_b -= learning_rate * self._grad_b
