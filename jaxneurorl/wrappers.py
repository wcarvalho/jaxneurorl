from typing import Optional, Tuple, Union, TypeVar

import jax
import jax.numpy as jnp
import chex

from functools import partial

from gymnax.environments import environment

from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents.basics import StepType


class EnvWrapper(object):
  """Base class for Gymnax wrappers."""

  def __init__(self, env):
    self._env = env

  # provide proxy access to regular attributes of wrapped object
  def __getattr__(self, name):
    return getattr(self._env, name)


class TimestepWrapper(EnvWrapper):
  """Flatten the observations of the environment."""

  def __init__(
    self,
    env: environment.Environment,
    autoreset: bool = True,
  ):
    super().__init__(env)
    self._autoreset = autoreset

  def reset(
    self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
  ) -> Tuple[TimeStep, dict]:
    obs, state = self._env.reset(key, params)
    # Get shape from first leaf of obs, assuming it's a batch dimension
    first_leaf = jax.tree_util.tree_leaves(obs)[0]
    shape = first_leaf.shape[0:1] if first_leaf.ndim > 1 else ()
    timestep = TimeStep(
      state=state,
      observation=obs,
      discount=jnp.ones(shape, dtype=jnp.float32),
      reward=jnp.zeros(shape, dtype=jnp.float32),
      step_type=jnp.full(shape, StepType.FIRST, dtype=StepType.FIRST.dtype),
    )
    return timestep

  def step(
    self,
    key: chex.PRNGKey,
    prior_timestep: TimeStep,
    action: Union[int, float],
    params: Optional[environment.EnvParams] = None,
  ) -> Tuple[TimeStep, dict]:
    def env_step(prior_timestep_):
      obs, state, reward, done, info = self._env.step(
        key, prior_timestep_.state, action, params
      )
      del info
      return TimeStep(
        state=state,
        observation=obs,
        discount=1.0 - done.astype(jnp.float32),
        reward=reward,
        step_type=jnp.where(done, StepType.LAST, StepType.MID),
      )

    if self._autoreset:
      # if prior was last, reset
      # otherwise, do regular step
      timestep = jax.lax.cond(
        prior_timestep.last(),
        lambda: self.reset(key, params),
        lambda: env_step(prior_timestep),
      )
    else:
      timestep = env_step(prior_timestep)
    return timestep
