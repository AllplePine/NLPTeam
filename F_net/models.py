'''编码模型'''


import functools
import math
from typing import Any, Dict, Tuple, Union, Optional

#Flax is a neural network library for JAX that is designed for flexibility.
from flax import linen as nn
from flax.training.common_utils import onehot
from jax import lax
from jax import random
from jax import numpy as jnp
import ml_collections
from scipy import linalg

from configs.base import ModelArchitecture, HybridAttentionLayout