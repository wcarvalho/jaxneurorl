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

    @nn.compact
    def __call__(self, obs: Observation):
        # [B, H, W, C]
        obs = jax.tree_map(lambda x: x.astype(jnp.float32), obs)
        image = nn.Sequential([
                # for symbolic input
                nn.Conv(64, (1, 1), strides=1, kernel_init=nn.initializers.truncated_normal(stddev=1.0)),
                # for future outputs
                nn.Conv(64, (4, 4), strides=2),
                nn.relu,
                nn.Conv(64, (3, 3), strides=1),
                nn.relu,
                nn.Conv(64, (3, 3), strides=1),
                nn.relu,
            ])(obs.image)
        image = image.reshape(image.shape[0], -1)

        # [B, D]
        vector_inputs = (obs.task_w, obs.state_features, obs.has_occurred, obs.pocket)
        vector_inputs = jnp.concatenate(vector_inputs, axis=-1)

        vector = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.truncated_normal(stddev=1.0))(vector_inputs)
        vector = nn.Dense(self.hidden_dim)(vector)

        outputs = jnp.concatenate((image, vector), axis=-1)

        return outputs