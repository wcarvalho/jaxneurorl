from typing import Optional

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import math

from projects.humansf.keyroom import Observation

def get_activation_fn(k: str):
    if k == 'relu': return nn.relu
    elif k == 'leaky_relu': return nn.leaky_relu
    elif k == 'tanh': return nn.tanh
    raise RuntimeError

class Block(nn.Module):
  features: int
  activation: str = 'relu'

  @nn.compact
  def __call__(self, x, _):
    x = nn.Dense(self.features, use_bias=False)(x)
    x = get_activation_fn(self.activation)(x)
    return x, None

class MLP(nn.Module):
  hidden_dim: int
  num_layers: int = 1
  out_dim: Optional[int] = None
  activation: str = 'relu'
  activate_final: bool = True

  @nn.compact
  def __call__(self, x):
    if self.num_layers == 0: return x
    for _ in range(self.num_layers-1):
        x, _ = Block(self.hidden_dim, self.activation)(x, None)

    x = nn.Dense(self.out_dim or self.hidden_dim, use_bias=False)(x)
    if self.activate_final:
       x = get_activation_fn(self.activation)(x)
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
    include_extras: bool = False

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
            embed(obs.local_position),  # binary
            embed(obs.pocket),          # binary
            embed(obs.prev_action),          # binary
        )
        if self.include_extras:
           vector_inputs += (
                embed(obs.state_features),  # binary
                embed(obs.has_occurred),    # binary
                embed(obs.position),        # binary
           )

        vector = jnp.concatenate(vector_inputs, axis=-1)
        vector = MLP(512, self.num_embed_layers)(vector)
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


class HouzemazeObsEncoder(nn.Module):
    """_summary_

    - observation encoder: CNN over binary inputs
    - MLP with truncated-normal-initialized Linear layer as initial layer for other inputs
    """
    embed_hidden_dim: int = 32
    embed_vector_dim: int = 512
    grid_hidden_dim: int = 256
    include_task: bool = True
    init: str = 'word_init'
    num_embed_layers: int = 1
    num_grid_layers: int = 1
    num_joint_layers: int = 1
    include_extras: bool = False

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

        def embed(x): return nn.Dense(
            self.embed_hidden_dim, kernel_init=initialization,
            use_bias=False)(x)

        # [B, D]
        vector_inputs = (
            embed(obs.direction),       # 1-hot
            embed(obs.position),  # binary
            embed(obs.prev_action),          # binary
        )
        if self.include_extras:
           vector_inputs += (
               embed(obs.state_features),  # binary
           )

        vector = jnp.concatenate(vector_inputs, axis=-1)
        vector = MLP(self.embed_vector_dim, self.num_embed_layers)(vector)
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

class CategoricalHouzemazeObsEncoder(nn.Module):
    """_summary_

    - observation encoder: CNN over binary inputs
    - MLP with truncated-normal-initialized Linear layer as initial layer for other inputs
    """
    num_categories: int
    include_task: bool = True
    embed_hidden_dim: int = 64
    mlp_hidden_dim: int = 256
    num_mlp_layers: int = 0
    activation: str = 'relu'

    @nn.compact
    def __call__(self, obs: Observation):
        activation = get_activation_fn(self.activation)
        flatten = lambda x: x.reshape(x.shape[0], -1)

        all_flattened = jnp.concatenate((
            flatten(obs.image),
            obs.direction[:, None],
            obs.position[:, None],
            obs.prev_action[:, None]),
            axis=-1
            ).astype(jnp.int32)
        embedding = nn.Embed(
            num_embeddings=self.num_categories,
            features=self.embed_hidden_dim,
            )(all_flattened)
        embedding = flatten(embedding)

        if self.include_task:
            task_w = nn.Dense(
                128)(obs.task_w.astype(jnp.float32))  # [continuous]
            outputs = (embedding, task_w)
            outputs = jnp.concatenate(outputs, axis=-1)
        else:
            outputs = embedding
        outputs = activation(outputs)
        outputs = MLP(
            self.mlp_hidden_dim,
            self.num_mlp_layers,
            activation=self.activation)(outputs)
        return outputs
