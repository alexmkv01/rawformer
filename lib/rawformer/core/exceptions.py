"""Custom exceptions for rawformer."""


class ForwardNotCalledError(RuntimeError):
    """Raised when backward() or update_params() is called before forward()."""

    def __init__(self, layer_name: str = "") -> None:
        prefix = f"{layer_name}: " if layer_name else ""
        super().__init__(f"{prefix}forward() must be called before backward()")


class ShapeMismatchError(ValueError):
    """Raised when input tensor shapes are incompatible."""

    def __init__(self, detail: str = "incompatible shapes") -> None:
        super().__init__(detail)
