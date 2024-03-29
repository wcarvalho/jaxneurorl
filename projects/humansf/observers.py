
import abc
import collections
from typing import Dict, Union, Optional, Callable

import jax
import jax.numpy as jnp
import flashbax as fbx
from flax import struct
import numpy as np
import wandb

from library.wrappers import TimeStep

Number = Union[int, float, np.float32, jnp.float32]


class Observer(abc.ABC):
  """An interface for collecting metrics/counters from actor and env."""

  @abc.abstractmethod
  def observe_first(
    self,
    first_timestep: TimeStep,
    agent_state: jax.Array) -> None:
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

  @abc.abstractmethod
  def flush_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""


@struct.dataclass
class BasicObserverState:
  episode_returns: jax.Array
  episode_lengths: jax.Array
  #episode_starts: jax.Array
  #action_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  #timestep_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  #prediction_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  task_info_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  idx: jax.Array = jnp.array(0, dtype=jnp.int32)
  episodes: jax.Array = jnp.array(0, dtype=jnp.int32)
  env_steps: jax.Array = jnp.array(0, dtype=jnp.int32)


def get_first(b): return jax.tree_map(lambda x:x[0], b)

def add_first_to_buffer(
    buffer: fbx.trajectory_buffer.TrajectoryBufferState,
    buffer_state: struct.PyTreeNode,
    x: struct.PyTreeNode):
  """
  x: [num_envs, ...]
  get first env data and dummy dummy time dim.
  """
  x = jax.tree_map(lambda y: y[:1, np.newaxis], x)
  return buffer.add(buffer_state, x)



class TaskObserver(Observer):
  """This is an observer that keeps track of timesteps, actions, and predictions.

  It only uses information from the first envionment. Annoying to track each env.

  """
  def __init__(
      self,
      log_period: int = 50_000,
      max_episode_length: int = 200,
      max_num_episodes: int = 200,
      extract_task_info: Callable[[TimeStep], struct.PyTreeNode] = None,
      get_task_name: Callable[[struct.PyTreeNode], str] = None,
      **kwargs,
      ):

    assert extract_task_info is not None
    assert get_task_name is not None

    self.get_task_name = get_task_name
    self.extract_task_info = extract_task_info

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

  def init(self,
           example_timestep,
           example_action,
           example_predictions):

    observer_state = BasicObserverState(
      episode_returns=jnp.zeros((self.log_period), dtype=jnp.float32),
      #episode_starts=jnp.zeros((self.log_period), dtype=jnp.int32),
      episode_lengths=jnp.zeros((self.log_period), dtype=jnp.int32),
      task_info_buffer=self.task_info_buffer.init(
         get_first(self.extract_task_info(example_timestep))),
      #timestep_buffer=self.buffer.init(get_first(example_timestep)),
      #action_buffer=self.buffer.init(get_first(example_action)),
      #prediction_buffer=self.buffer.init(get_first(example_predictions)),
    )
    return observer_state

  def observe_first(
    self,
    observer_state: BasicObserverState,
    first_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
    ) -> BasicObserverState:

    del agent_state

    task_info_buffer = add_first_to_buffer(
      buffer=self.buffer,
      buffer_state=observer_state.task_info_buffer,
      x=self.extract_task_info(first_timestep),
    )

    observer_state = observer_state.replace(
      task_info_buffer=task_info_buffer)

    return observer_state

  def observe(
    self,
    observer_state: BasicObserverState,
    predictions: struct.PyTreeNode,
    action: jax.Array,
    next_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
    **kwargs,
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

    def update_episode(os):
      # within same episode

      # update return/length information
      idx = os.idx
      # update observer state
      return os.replace(
        episode_lengths=os.episode_lengths.at[idx].add(1),
        episode_returns=os.episode_returns.at[idx].add(first_next_timestep.reward),
      )

    def advance_episode(os):
      # beginning new episode
      idx = os.idx + 1
      task_info_buffer = add_first_to_buffer(
            buffer=self.buffer,
            buffer_state=os.task_info_buffer,
            x=self.extract_task_info(next_timestep),
          )

      return os.replace(
        idx=idx,
        task_info_buffer=task_info_buffer,
        episode_lengths=os.episode_lengths.at[idx].add(1),
        episode_returns=os.episode_returns.at[idx].add(first_next_timestep.reward)
        )

    observer_state = jax.lax.cond(
      first_next_timestep.first(),
      advance_episode,
      update_episode,
      observer_state
    )

    return observer_state

  def flush_metrics(
    self,
    key: str,
    observer_state: BasicObserverState,
    force: bool = False,
    shared_metrics: dict = {}):
    def callback(os: BasicObserverState, sm: dict):
        if wandb.run is not None:
          end = os.idx + 1

          if not force:
            if not end % self.log_period == 0: return

          metrics = collections.defaultdict(list)

          return_key = lambda name: f'{key}/0. {name.capitalize()} - AvgReturn'
          length_key = lambda name: f'{key}/1. {name.capitalize()} - AvgLength'

          for idx in range(end):
            task_info = jax.tree_map(lambda x: x[0, idx], os.task_info_buffer.experience)
            task_name = self.get_task_name(task_info)

            metrics[return_key(task_name)].append(os.episode_returns[idx])
            metrics[length_key(task_name)].append(os.episode_lengths[idx])

          # get average of all values
          metrics = {k: np.array(v).mean() for k, v in metrics.items()}

          # update with shared metrics
          metrics.update({f'{key}/{k}': v for k,v in sm.items()})
          wandb.log(metrics)
    jax.debug.callback(callback, observer_state, shared_metrics)
