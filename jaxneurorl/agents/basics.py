from flax import struct
import jax
import jax.numpy as jnp


class StepType(jnp.uint8):
    FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
    MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
    LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)


class TimeStep(struct.PyTreeNode):
    state: struct.PyTreeNode

    step_type: StepType
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST