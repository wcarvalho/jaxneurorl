import abc

from typing import Dict, Union, Optional

import jax
import jax.numpy as jnp
import flashbax as fbx
from flax import struct
from flax.struct import field

import numpy as np
import wandb

from jaxneurorl.agents.basics import TimeStep

Number = Union[int, float, np.float32, jnp.float32]


class Observer(abc.ABC):
  """An interface for collecting metrics/counters from actor and env."""

  @abc.abstractmethod
  def observe_first(self, first_timestep: TimeStep, agent_state: jax.Array) -> None:
    """Observes the initial state and initial time-step.

    Usually state will be all zeros and time-step will be output of reset."""

  @abc.abstractmethod
  def observe(
    self,
    predictions: struct.PyTreeNode,
    action: jax.Array,
    next_timestep: TimeStep,
  ) -> None:
    """Observe state and action that are due to observation of time-step.

    Should be state after previous time-step along"""


@struct.dataclass
class BasicObserverState:
  episode_returns: jax.Array
  episode_lengths: jax.Array
  episode_starts: jax.Array
  # action_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  # timestep_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  # prediction_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  task_info_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  idx: jax.Array = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
  episodes: jax.Array = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
  env_steps: jax.Array = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))


def add_to_buffer(
  buffer: fbx.trajectory_buffer.TrajectoryBufferState,
  buffer_state: struct.PyTreeNode,
  x: struct.PyTreeNode,
):
  x = jax.tree.map(lambda y: y[:, np.newaxis], x)
  return buffer.add(buffer_state, x)


def get_first(b):
  return jax.tree.map(lambda x: x[0], b)


def add_first_to_buffer(
  buffer: fbx.trajectory_buffer.TrajectoryBufferState,
  buffer_state: struct.PyTreeNode,
  x: struct.PyTreeNode,
):
  """
  x: [num_envs, ...]
  get first env data and dummy dummy time dim.
  """
  x = jax.tree.map(lambda y: y[:1, np.newaxis], x)
  return buffer.add(buffer_state, x)


class BasicObserverOld(Observer):
  """This is an observer that keeps track of timesteps, actions, and predictions.

  It only uses information from the first envionment. Annoying to track each env.

  """

  def __init__(
    self,
    log_period: int = 50_000,
    max_episode_length: int = 200,
    max_num_episodes: int = 200,
    **kwargs,
  ):
    self.log_period = log_period
    self.max_episode_length = max_episode_length
    self.buffer = fbx.make_trajectory_buffer(
      max_length_time_axis=self.max_episode_length,
      min_length_time_axis=1,
      sample_batch_size=1,  # unused
      add_batch_size=1,
      sample_sequence_length=1,  # unused
      period=1,
    )
    self.task_info_buffer = fbx.make_trajectory_buffer(
      max_length_time_axis=max_num_episodes,
      min_length_time_axis=1,
      sample_batch_size=1,  # unused
      add_batch_size=1,
      sample_sequence_length=1,  # unused
      period=1,
    )

  def init(self, example_timestep, example_action, example_predictions):
    observer_state = BasicObserverState(
      episode_returns=jnp.zeros((self.log_period), dtype=jnp.float32),
      episode_starts=jnp.zeros((self.log_period), dtype=jnp.int32),
      episode_lengths=jnp.zeros((self.log_period), dtype=jnp.int32),
      task_info_buffer=self.task_info_buffer.init(get_first(example_timestep)),
      # timestep_buffer=self.buffer.init(get_first(example_timestep)),
      # action_buffer=self.buffer.init(get_first(example_action)),
      # prediction_buffer=self.buffer.init(get_first(example_predictions)),
    )
    return observer_state

  def observe_first(
    self,
    observer_state: BasicObserverState,
    first_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
  ) -> BasicObserverState:
    del agent_state

    # timestep_buffer = add_to_buffer(
    #  buffer=self.buffer,
    #  buffer_state=observer_state.timestep_buffer,
    #  x=first_timestep,
    # )

    # observer_state = observer_state.replace(timestep_buffer=timestep_buffer)

    return observer_state

  def observe(
    self,
    observer_state: BasicObserverState,
    predictions: struct.PyTreeNode,
    action: jax.Array,
    next_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
    key: str = "actor",
    maybe_flush: bool = False,
    maybe_reset: bool = False,
  ) -> None:
    """Update log and flush if terminal + log period hit.


    Args:
        agent_state (struct.PyTreeNode): _description_
        predictions (struct.PyTreeNode): _description_
        action (jax.Array): _description_
        next_timestep (TimeStep): _description_
    """
    del agent_state

    ## update buffer with information
    # timestep_buffer = add_to_buffer(
    #  buffer=self.buffer,
    #  buffer_state=observer_state.timestep_buffer,
    #  x=next_timestep,
    # )

    # prediction_buffer = add_to_buffer(
    #  buffer=self.buffer,
    #  buffer_state=observer_state.prediction_buffer,
    #  x=predictions,
    # )

    # action_buffer = add_to_buffer(
    #  buffer=self.buffer,
    #  buffer_state=observer_state.action_buffer,
    #  x=action,
    # )

    # only use first time-step
    next_timestep = jax.tree.map(lambda x: x[0], next_timestep)

    # update return/length information
    idx = observer_state.idx
    episode_returns = observer_state.episode_returns.at[idx].add(next_timestep.reward)
    episode_lengths = observer_state.episode_lengths.at[idx].add(1)
    total_env_steps = observer_state.env_steps + 1

    # is_last = next_timestep.last()
    # is_last_int = is_last.astype(jnp.int32)

    # update observer state
    observer_state = observer_state.replace(
      # timestep_buffer=timestep_buffer,
      # prediction_buffer=prediction_buffer,
      # action_buffer=action_buffer,
      episode_returns=episode_returns,
      episode_lengths=episode_lengths,
      env_steps=total_env_steps,
      # episodes=observer_state.episodes + is_last_int
    )

    #############
    # flushing and logging
    # it's easier to just do 1 env, so will ignore rest...
    # otherwise, having competing conditons when when episode ends
    #############
    first_next_timestep = jax.tree.map(lambda x: x[0], next_timestep)

    ##-----------------------
    ## if final time-step and log-period has been hit, flush the metrics
    ##-----------------------
    # if maybe_flush:
    #  flush = jnp.logical_and(
    #    idx % self.log_period == 0,
    #    first_next_timestep.last(),
    #    )

    #  jax.lax.cond(
    #      flush,
    #      lambda: self.flush_metrics(key, observer_state),
    #      lambda: None,
    #  )

    # -----------------------
    # if final time-step, reset buffers. only keep around 1.
    # -----------------------
    if maybe_reset:
      observer_state = jax.lax.cond(
        first_next_timestep.last(),
        lambda: self.reset_buffers(observer_state),
        lambda: observer_state,
      )

    return observer_state

  def reset_buffers(self, observer_state: BasicObserverState):
    """Reset all the buffers in the observer-state (presumably at end of episode)."""

    # first env + first time-step
    get_first_instance = lambda o: get_first(get_first(o))

    timestep_buffer = get_first_instance(observer_state.timestep_buffer.experience)
    action_buffer = get_first_instance(observer_state.action_buffer.experience)
    prediction_buffer = get_first_instance(observer_state.prediction_buffer.experience)

    return observer_state.replace(
      timestep_buffer=self.buffer.init(timestep_buffer),
      action_buffer=self.buffer.init(action_buffer),
      prediction_buffer=self.buffer.init(prediction_buffer),
      idx=observer_state.idx + 1,
    )

  def flush_metrics(
    self,
    key: str,
    observer_state: BasicObserverState,
    force: bool = False,
    shared_metrics: dict = {},
  ):
    def callback(os: BasicObserverState, sm):
      if wandb.run is not None:
        idx = os.idx + 1

        if not force:
          if not idx % self.log_period == 0:
            return

        metrics = {
          f"{key}/avg_episode_length": os.episode_lengths[0, :idx].mean(),
          f"{key}/avg_episode_return": os.episode_returns[0, :idx].mean(),
          # f'{key}/num_episodes': os.episodes.sum(),
          # f'{key}/num_steps': os.env_steps.sum(),
        }
        metrics.update({f"{key}/{k}": v for k, v in sm.items()})
        wandb.log(metrics)

    jax.debug.callback(callback, observer_state, shared_metrics)


class BasicObserver(Observer):
  """This is an observer that keeps track of timesteps, actions, and predictions.

  It only uses information from the first envionment. Annoying to track each env.

  """

  def __init__(
    self,
    log_period: int = 50_000,
    max_episode_length: int = 200,
    max_num_episodes: int = 200,
    **kwargs,
  ):
    self.log_period = log_period
    self.max_episode_length = max_episode_length
    self.buffer = fbx.make_trajectory_buffer(
      max_length_time_axis=self.max_episode_length,
      min_length_time_axis=1,
      sample_batch_size=1,  # unused
      add_batch_size=1,
      sample_sequence_length=1,  # unused
      period=1,
    )
    self.task_info_buffer = fbx.make_trajectory_buffer(
      max_length_time_axis=max_num_episodes,
      min_length_time_axis=1,
      sample_batch_size=1,  # unused
      add_batch_size=1,
      sample_sequence_length=1,  # unused
      period=1,
    )

  def init(self, example_timestep, example_action, example_predictions):
    observer_state = BasicObserverState(
      episode_returns=jnp.zeros((self.log_period), dtype=jnp.float32),
      episode_starts=jnp.zeros((self.log_period), dtype=jnp.int32),
      episode_lengths=jnp.zeros((self.log_period), dtype=jnp.int32),
      task_info_buffer=self.task_info_buffer.init(get_first(example_timestep)),
      # timestep_buffer=self.buffer.init(get_first(example_timestep)),
      # action_buffer=self.buffer.init(get_first(example_action)),
      # prediction_buffer=self.buffer.init(get_first(example_predictions)),
    )
    return observer_state

  def observe_first(
    self,
    observer_state: BasicObserverState,
    first_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
  ) -> BasicObserverState:
    del agent_state

    # timestep_buffer = add_to_buffer(
    #  buffer=self.buffer,
    #  buffer_state=observer_state.timestep_buffer,
    #  x=first_timestep,
    # )

    # observer_state = observer_state.replace(timestep_buffer=timestep_buffer)

    return observer_state

  def observe(
    self,
    observer_state: BasicObserverState,
    predictions: struct.PyTreeNode,
    action: jax.Array,
    next_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
    key: str = "actor",
    maybe_flush: bool = False,
    maybe_reset: bool = False,
  ) -> None:
    """Update log and flush if terminal + log period hit.


    Args:
        agent_state (struct.PyTreeNode): _description_
        predictions (struct.PyTreeNode): _description_
        action (jax.Array): _description_
        next_timestep (TimeStep): _description_
    """
    del agent_state

    # only use first time-step
    first_next_timestep = get_first(next_timestep)

    def advance_episode(os):
      # beginning new episode
      idx = os.idx + 1
      return os.replace(
        idx=idx,
        episode_lengths=os.episode_lengths.at[idx].add(1),
        episode_returns=os.episode_returns.at[idx].add(first_next_timestep.reward),
      )

    def update_episode(os):
      # update return/length information
      idx = os.idx
      # update observer state
      return os.replace(
        episode_lengths=os.episode_lengths.at[idx].add(1),
        episode_returns=os.episode_returns.at[idx].add(first_next_timestep.reward),
      )

    observer_state = jax.lax.cond(
      first_next_timestep.first(), advance_episode, update_episode, observer_state
    )

    return observer_state

  def flush_metrics(
    self,
    key: str,
    observer_state: BasicObserverState,
    force: bool = False,
    shared_metrics: dict = {},
  ):
    def callback(os: BasicObserverState, sm):
      if wandb.run is not None:
        idx = min(os.idx + 1, self.log_period)

        if not force:
          if not idx % self.log_period == 0:
            return

        metrics = {
          f"{key}/avg_episode_length": os.episode_lengths[:idx].mean(),
          f"{key}/avg_episode_return": os.episode_returns[:idx].mean(),
          # f'{key}/num_episodes': os.episodes.sum(),
          # f'{key}/num_steps': os.env_steps.sum(),
        }
        metrics.update({f"{key}/{k}": v for k, v in sm.items()})
        wandb.log(metrics)

    jax.debug.callback(callback, observer_state, shared_metrics)
