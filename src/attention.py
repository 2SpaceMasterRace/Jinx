import math
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def scaled_dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    scale: float | None,
    mask: ArrayLike | None,
) -> Array:
    """
    Implements the scaled dot product attention function

    query is the query matrix,
    key is the key matrix,
    value is the value matrix,
    scale is the dimension of each individual query and key, (typically 1 / sqrt(d)),
    mask is the  mask matrix
    """
    d_k = query.shape[-1]
    transpose_key: ArrayLike = jnp.matrix_transpose(key)
    score = jnp.matmul(query, transpose_key) / math.sqrt(d_k)
