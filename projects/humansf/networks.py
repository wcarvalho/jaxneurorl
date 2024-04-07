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
    init: str = 'word_embed'

    @nn.compact
    def __call__(self, obs: Observation):
        obs = jax.tree_map(lambda x: x.astype(jnp.float32), obs)

        ###################
        # embed vector inputs
        ###################
        word_init = nn.initializers.variance_scaling(
                1.0, 'fan_in', 'normal', out_axis=0)
        embed = lambda x: nn.Dense(
            self.hidden_dim, kernel_init=word_init,
            use_bias=False)(x)

        # [B, D]
        vector_inputs = (
            embed(obs.task_w),
            embed(obs.direction),
            embed(obs.state_features),
            embed(obs.has_occurred),
            embed(obs.local_position),
            embed(obs.position),
            embed(obs.pocket),
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
            kernel_init=word_init, use_bias=False)(image)

        # turn into vector
        image = image.reshape(image.shape[0], -1)

        ###################
        # combine
        ###################

        outputs = jnp.concatenate((image, vector), axis=-1)
        outputs = nn.Sequential([
            nn.Dense(self.image_hidden_dim), nn.relu,
            nn.Dense(self.image_hidden_dim)
        ])(outputs)

        return outputs
