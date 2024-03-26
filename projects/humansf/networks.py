import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np

from keyroom import Observation

class KeyroomObsEncoder(nn.Module):
    """_summary_

    - observation encoder: CNN
    """
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs: Observation):
        # [B, H, W, C]
        image = nn.Sequential(
            [
                nn.Conv(16, (2, 2), padding="VALID",
                        kernel_init=orthogonal(np.sqrt(2))),
                nn.relu,
                nn.Conv(32, (2, 2), padding="VALID",
                        kernel_init=orthogonal(np.sqrt(2))),
                nn.relu,
                nn.Conv(64, (2, 2), padding="VALID",
                        kernel_init=orthogonal(np.sqrt(2))),
            ]
        )(obs.image)
        image = image.reshape(image.shape[0], -1)

        # [B, D]
        vector_inputs = (obs.task_w, obs.state_features, obs.has_occurred, obs.pocket)
        vector_inputs = jnp.concatenate(vector_inputs, axis=-1)
        vector = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.truncated_normal(stddev=1.0))(vector_inputs)
        vector = nn.Dense(self.hidden_dim)(jax.nn.relu(vector))

        outputs = jnp.concatenate((image, vector), axis=-1)

        return outputs