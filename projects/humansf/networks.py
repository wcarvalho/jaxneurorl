from typing import Optional

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import math

from projects.humansf.keyroom import Observation

from jaxneurorl.agents.value_based_pqn import MLP, BatchRenorm, get_activation_fn

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
    num_embed_layers: int = 0
    norm_type: str = "none"
    activation: str = 'relu'

    @nn.compact
    def __call__(self, obs: Observation, train: bool = False):
        has_batch = obs.image.ndim == 3
        assert obs.image.ndim in (2, 3), 'either [B, H, W] or [H, W]'
        if has_batch:
            flatten = lambda x: x.reshape(x.shape[0], -1)
            expand = lambda x: x[:, None]
        else:
            flatten = lambda x: x.reshape(-1)
            expand = lambda x: x[None]

        act = get_activation_fn(self.activation)
        if self.norm_type == 'layer_norm':
            norm = lambda x: act(nn.LayerNorm()(x))
        elif self.norm_type == 'batch_norm':
            norm = lambda x: act(BatchRenorm(use_running_average=not train)(x))
        elif self.norm_type == 'none':
            norm = lambda x: x
        else:
            raise NotImplementedError(self.norm_type)

        all_flattened = jnp.concatenate((
            flatten(obs.image),
            obs.position,
            expand(obs.direction),
            expand(obs.prev_action)),
            axis=-1
            ).astype(jnp.int32)
        embedding = nn.Embed(
            num_embeddings=self.num_categories,
            features=self.embed_hidden_dim,
            )(all_flattened)
        embedding = flatten(embedding)
        embedding = norm(embedding)
        embedding = MLP(
            self.mlp_hidden_dim,
            self.num_embed_layers,
            norm_type=self.norm_type,
            activation=self.activation)(embedding)

        if self.include_task:
            kernel_init = nn.initializers.variance_scaling(
                1.0, 'fan_in', 'normal', out_axis=0)
            task_w = nn.Dense(
                128, kernel_init=kernel_init,
                )(obs.task_w.astype(jnp.float32))
            task_w = norm(task_w)
            outputs = (embedding, task_w)
            outputs = jnp.concatenate(outputs, axis=-1)
        else:
            outputs = embedding

        outputs = MLP(
            self.mlp_hidden_dim,
            self.num_mlp_layers,
            norm_type=self.norm_type,
            activation=self.activation)(outputs)

        return outputs
