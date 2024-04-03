import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp

from projects.humansf.keyroom import Observation

class KeyroomObsEncoder(nn.Module):
    """_summary_

    - observation encoder: CNN over binary inputs
    - MLP with truncated-normal-initialized Linear layer as initial layer for other inputs
    """
    hidden_dim: int = 128
    conv_dim: int = 16
    init: str = 'word_embed'

    @nn.compact
    def __call__(self, obs: Observation):
        obs = jax.tree_map(lambda x: x.astype(jnp.float32), obs)

        # [B, H, W, C]
        image = obs.image
        # [B, D]
        vector_inputs = (obs.task_w, obs.state_features, obs.has_occurred, obs.pocket)
        vector = jnp.concatenate(vector_inputs, axis=-1)

        if self.init == 'word_embed':
            kernel_init = nn.initializers.truncated_normal(stddev=1.0)
            image = nn.Conv(
                self.conv_dim, (1, 1), strides=1,
                kernel_init=kernel_init)
        elif self.init == 'default':
            kernel_init = nn.initializers.lecun_normal()
        else:
            raise NotImplementedError(self.init)
        import ipdb; ipdb.set_trace()

        image = nn.Sequential([
                # for future outputs
                nn.Conv(self.conv_dim, (4, 4), strides=2),
                nn.relu,
                nn.Conv(self.conv_dim, (3, 3), strides=1),
            ])(image)

        vector = nn.Dense(self.hidden_dim, kernel_init=kernel_init)

        image = image.reshape(image.shape[0], -1)
        outputs = jnp.concatenate((image, vector), axis=-1)

        return outputs