"""Fourier Transforms used by FNet."""

import functools

import jax
from jax import lax
import jax.numpy as jnp


def two_dim_matmul(
    x,
    matrix_dim_one,
    matrix_dim_two,
    precision = lax.Precision.DEFAULT):
  """Applies 2D matrix multiplication to 2D input arrays.

  Args:
    x: Input of shape [MAX_SEQ_LEN, HIDDEN_DIM]
    matrix_dim_one: [MAX_SEQ_LEN, MAX_SEQ_LEN] matrix to apply to first
      (sequence) dimension of input.
    matrix_dim_two: [HIDDEN_DIM, HIDDEN_DIM] matrix to apply to second (hidden)
      dimension of input.
    precision: XLA precision for matrix multiplication operation.

  Returns:
    [MAX_SEQ_LEN, HIDDEN_DIM] array resulting from application of two
      consecutive matrix multiplications.
  """
  return _two_dim_matmul(x, matrix_dim_one, matrix_dim_two, precision)


@functools.partial(jax.jit, static_argnums=3)
def _two_dim_matmul(x, matrix_dim_one,
                    matrix_dim_two,
                    precision):
  """Applies 2D matrix multiplication to 2D input arrays."""
  return jnp.einsum(  # pytype: disable=wrong-arg-types  # jnp-type
      "ij,jk,ni->nk",
      x,
      matrix_dim_two,
      matrix_dim_one,
      optimize=True,
      precision=precision)