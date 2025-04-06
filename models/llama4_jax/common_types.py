# <Apache-2.0 extracted from AI-Hypercomputer/maxtext>

import os.path

import jax.numpy as jnp
from aqt.jax.v2 import aqt_tensor

Array = jnp.ndarray
DType = jnp.dtype
KVTensor = aqt_tensor.QTensor

# </Apache-2.0 extracted from AI-Hypercomputer/maxtext>

# <Apache-2.0 extracted from keras-team/keras>

# The type of float to use throughout a session.
_FLOATX = "float32"

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = "channels_last"

# Default backend: TensorFlow.
_BACKEND = "tensorflow"


def floatx():
    """Return the default float type, as a string.

    E.g. `'bfloat16'`, `'float16'`, `'float32'`, `'float64'`.

    Returns:
        String, the current default float type.

    Example:

    >>> keras.config.floatx()
    'float32'

    """
    return _FLOATX


def set_floatx(value):
    """Set the default float dtype.

    Note: It is not recommended to set this to `"float16"` for training,
    as this will likely cause numeric stability issues.
    Instead, mixed precision, which leverages
    a mix of `float16` and `float32`.

    Args:
        value: String; `'bfloat16'`, `'float16'`, `'float32'`, or `'float64'`.

    Examples:
    >>> common_types.floatx()
    'float32'

    >>> common_types.set_floatx('float64')
    >>> common_types.floatx()
    'float64'

    >>> # Set it back to float32
    >>> common_types.set_floatx('float32')

    Raises:
        ValueError: In case of invalid value.
    """
    global _FLOATX
    accepted_dtypes = {"bfloat16", "float16", "float32", "float64"}
    if value not in accepted_dtypes:
        raise ValueError(
            f"Unknown `floatx` value: {value}. "
            f"Expected one of {accepted_dtypes}"
        )
    _FLOATX = str(value)

# </Apache-2.0 extracted from keras-team/keras>

JAX_TARGET_DEVICE = os.environ.get("JAX_TARGET_DEVICE", "gpu")

# Final to add to dunder

__all__ = ["Array", "DType", "KVTensor", "floatx", "set_floatx"]
