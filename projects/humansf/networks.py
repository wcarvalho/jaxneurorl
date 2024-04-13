import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import math

from projects.humansf.keyroom import Observation


class KeyroomObsEncoder(nn.Module):
    """_summary_

    - observation encoder: CNN over binary inputs
    - MLP with truncated-normal-initialized Linear layer as initial layer for other inputs
    """
    hidden_dim: int = 128
    image_hidden_dim: int = 512
    init: str = 'word_init'

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
            self.hidden_dim, kernel_init=initialization,
            use_bias=False)(x)

        # [B, D]
        vector_inputs = (
            nn.Dense(self.hidden_dim, kernel_init=initialization)(obs.task_w),  # [continuous]
            embed(obs.direction),       # 1-hot
            embed(obs.state_features),  # binary
            embed(obs.has_occurred),    # binary
            embed(obs.local_position),  # binary
            embed(obs.position),        # binary
            embed(obs.pocket),          # binary
            embed(obs.prev_action),          # binary
        )
        vector = jnp.concatenate(vector_inputs, axis=-1)

        ###################
        # embed image inputs
        ###################
        # [B, H, W, C]
        image = obs.image
        assert image.ndim == 4
        image = nn.Conv(
            self.hidden_dim, (1, 1),
            kernel_init=initialization, use_bias=False)(image)

        # turn into vector
        image = image.reshape(image.shape[0], -1)

        ###################
        # combine
        ###################

        outputs = jnp.concatenate((image, vector), axis=-1)
        outputs = nn.Sequential([
            nn.Dense(self.image_hidden_dim), nn.relu,
            nn.Dense(self.image_hidden_dim), nn.relu
        ])(outputs)

        return outputs
