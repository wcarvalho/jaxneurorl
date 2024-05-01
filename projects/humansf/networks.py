from typing import Optional

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import math

from projects.humansf.keyroom import Observation

class Block(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x, _):
    x = nn.Dense(self.features, use_bias=False)(x)
    x = jax.nn.relu(x)
    return x, None

class MLP(nn.Module):
  hidden_dim: int
  num_layers: int = 1
  out_dim: Optional[int] = None
  activate_final: bool = True

  @nn.compact
  def __call__(self, x):
    if self.num_layers == 0: return x

    for _ in range(self.num_layers-1):
        x, _ = Block(self.hidden_dim)(x, None)

    x = nn.Dense(self.out_dim or self.hidden_dim, use_bias=False)(x)
    if self.activate_final:
       x = nn.relu(x)
    return x

class KeyroomObsEncoder(nn.Module):
    """_summary_

    - observation encoder: CNN over binary inputs
    - MLP with truncated-normal-initialized Linear layer as initial layer for other inputs
    """
    embed_hidden_dim: int = 32
    grid_hidden_dim: int = 256
    include_task: bool = True
    init: str = 'word_init'
    num_embed_layers: int = 1
    num_grid_layers: int = 1
    num_joint_layers: int = 1

    @nn.compact
    def __call__(self, obs: Observation):
        obs = jax.tree_map(lambda x: x.astype(jnp.float32), obs)

        ###################
        # embed vector inputs
        ###################
        if self.init == 'word_init':
            initialization = nn.initializers.variance_scaling(
                1.0, 'fan_in', 'normal', out_axis=0)
        elif self.init == 'word_init2':
            initialization = nn.initializers.variance_scaling(
                1.0, 'fan_in', 'normal', out_axis=-1)
        elif self.init == 'truncated':
            initialization = nn.initializers.truncated_normal(1.0)
        else:
            raise NotImplementedError

        embed = lambda x: nn.Dense(
            self.embed_hidden_dim, kernel_init=initialization,
            use_bias=False)(x)

        # [B, D]
        vector_inputs = (
            embed(obs.direction),       # 1-hot
            #embed(obs.state_features),  # binary
            #embed(obs.has_occurred),    # binary
            embed(obs.local_position),  # binary
            #embed(obs.position),        # binary
            embed(obs.pocket),          # binary
            embed(obs.prev_action),          # binary
        )
        vector = jnp.concatenate(vector_inputs, axis=-1)
        vector = MLP(128, self.num_embed_layers)(vector)
        ###################
        # embed image inputs
        ###################
        # [B, H, W, C]
        grid = obs.image
        assert grid.ndim in (3, 4)
        grid = nn.Conv(
            self.embed_hidden_dim, (1, 1),
            kernel_init=initialization, use_bias=False)(grid)

        # turn into vector
        if grid.ndim == 4:
            grid = grid.reshape(grid.shape[0], -1)
        elif grid.ndim == 3:
            grid = grid.reshape(-1)
        else:
           raise NotImplementedError
        grid = MLP(self.grid_hidden_dim,
                   self.num_grid_layers)(grid)

        if grid.ndim == 3:
           import ipdb; ipdb.set_trace()
        ###################
        # combine
        ###################
        if self.include_task:
            task_w = nn.Dense(
                128, kernel_init=initialization)(obs.task_w)  # [continuous]
            outputs = (grid, vector, task_w)
            outputs = jnp.concatenate(outputs, axis=-1)
        else:
            outputs = jnp.concatenate((grid, vector), axis=-1)

        outputs = MLP(
           self.grid_hidden_dim,
           self.num_joint_layers)(outputs)

        return outputs
