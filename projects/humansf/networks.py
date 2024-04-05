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

        # [B, H, W, C]
        image = obs.image
        assert image.ndim == 4
        # [B, D]
        #vector_inputs = (obs.task_w, obs.state_features, obs.has_occurred, obs.pocket)
        vector_inputs = (
            obs.task_w,
            obs.direction,
            obs.local_position,
            obs.position,
            obs.pocket,
        )
        vector = jnp.concatenate(vector_inputs, axis=-1)
        if self.init == 'word_embed':
            kernel_init = nn.initializers.truncated_normal(stddev=1.0)
            bias_init = constant(0.0)
            use_bias=True
        elif self.init == 'word_embed2':
            kernel_init = nn.initializers.variance_scaling(
                1.0, 'fan_in', 'normal', out_axis=0
            )
            bias_init = constant(0.0)
            use_bias=False
        elif self.init == 'jaxrl':
            kernel_init = nn.initializers.orthogonal(math.sqrt(2))
            bias_init = constant(0.0)
            use_bias=True
        elif self.init == 'default':
            kernel_init = nn.initializers.lecun_normal()
            bias_init = constant(0.0)
            use_bias=True
        else:
            raise NotImplementedError(self.init)

        #image = nn.Sequential([
        #    nn.Conv(128, (3, 3), strides=2, kernel_init=kernel_init, padding="VALID"),
        #    nn.relu,
        #    nn.Conv(128, (3, 3), strides=2, kernel_init=kernel_init, padding="VALID"),
        #    nn.relu,
        #])(image)
        assert image.shape[1] > 0, f'shape: {image.shape}!'
        image = image.reshape(image.shape[0], -1)

        image = nn.Dense(self.image_hidden_dim,
                         kernel_init=kernel_init,
                         bias_init=bias_init,
                         use_bias=use_bias)(image)
        vector = nn.Dense(self.hidden_dim,
                          kernel_init=kernel_init,
                          bias_init=bias_init,
                          use_bias=use_bias)(vector)

        outputs = jnp.concatenate((image, vector), axis=-1)

        return outputs
