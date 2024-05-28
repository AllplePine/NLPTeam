'''傅里叶变换'''

import functools

import jax
from jax import lax
import jax.numpy as jnp

def two_dim_matmul(
        x,
        matrix_dim_one,
        matric_dim_two,
        precision = lax.Precision.DEFAULT):
    '''2D矩阵乘法'''
    return _two_dim_matmul(x, matrix_dim_one, matric_dim_two, precision)

@functools.partial(jax.jit, static_argnums=3)
def _two_dim_matmul(
                x,
                matrix_dim_one,
                matrix_dim_two,
                precision):
        return jnp.einsum(
               "ij,jk,ni->nk",
               x,
               matrix_dim_one,
               matrix_dim_two,
               optimize=True,
               precision=precision)