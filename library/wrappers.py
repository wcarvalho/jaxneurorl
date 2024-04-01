from typing import Optional, Tuple, Union, TypeVar

import jax
import jax.numpy as jnp
import chex

from functools import partial

from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from singleagent.basics import TimeStep
from singleagent.basics import StepType


EnvCarryT = TypeVar("EnvCarryT")

class TimestepWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(
            self,
            env: environment.Environment,
            autoreset: bool = True,
            ):
        super().__init__(env)
        self._autoreset = autoreset

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        params: Optional[environment.EnvParams] = None
    ) -> Tuple[TimeStep, dict]:
        obs, state = self._env.reset(key, params)
        timestep = TimeStep(
            state=state,
            observation=obs,
            discount=0.0,
            reward=0.0,
            step_type=StepType.FIRST,
        )
        return timestep

    @partial(jax.jit, static_argnums=(0,))
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
              discount=1. - done.astype(jnp.float32),
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
