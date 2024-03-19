
import abc

from typing import Dict, Union, Optional

import jax
import jax.numpy as jnp
import flashbax as fbx
from flax import struct
import numpy as np
import wandb

from library.wrappers import TimeStep

Number = Union[int, float, np.float32, jnp.float32]


class Observer(abc.ABC):
  """An interface for collecting metrics/counters from actor and env.

  # Note: agent.state has as a field, the agents predictions, e.g. Q-values.

  Episode goes:
    # observe initial
    timestep = env.reset()
    state = agent.init_state()

    agent.observe_first(timestep)  # self.state is initialized
    # inside, self.observer.observe_first(state, timestep) is called

    while timestep.not_last():
      action = agent.select_action(timestep)  # agent.state is updated
      # inside, self.observer.observe_action(state, action) is called

      next_timestep = env.step(action)
      agent.observe(action, next_timestep)
      # inside, self.observer.observe_timestep(next_timestep) is called
      
      timestep = next_timestep

  """

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
  action_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  timestep_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  prediction_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  episodes: int
  env_steps: int


def add_to_buffer(
    buffer: fbx.trajectory_buffer.TrajectoryBufferState,
    buffer_state: struct.PyTreeNode,
    x: struct.PyTreeNode):
  x = jax.tree_map(
      lambda y: y[:, np.newaxis], x)
  return buffer.add(buffer_state, x)

def get_first(b): return jax.tree_map(lambda x:x[0], b)

class BasicObserver(Observer):
  """This is an observer that keeps track of timesteps, actions, and predictions.

  It 

  1. episode returns
  2. episode lengths

  Args:
      Observer (_type_): _description_
  """
  def __init__(
      self,
      num_envs: int,
      log_period: int = 1000,
      max_episode_length: int = 100,
      ):

    self.num_envs = num_envs
    self.log_period = log_period
    self.max_episode_length = max_episode_length
    self.buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=self.max_episode_length,
        min_length_time_axis=1,
        sample_batch_size=1,  # unused
        add_batch_size=num_envs,
        sample_sequence_length=1,  # unused
        period=1,
    )

  def init(self,
           example_timestep,
           example_action,
           example_predictions):

    observer_state = BasicObserverState(
      episode_returns=jnp.zeros((self.num_envs, self.log_period), dtype=jnp.float32),
      episode_lengths=jnp.zeros((self.num_envs, self.log_period), dtype=jnp.int32),
      timestep_buffer=self.buffer.init(get_first(example_timestep)),
      action_buffer=self.buffer.init(get_first(example_action)),
      prediction_buffer=self.buffer.init(get_first(example_predictions)),
      episodes=jnp.zeros(self.num_envs, dtype=jnp.int32),
      env_steps=jnp.zeros(self.num_envs, dtype=jnp.int32),
      )
    return observer_state

  def observe_first(
    self,
    observer_state: BasicObserverState,
    first_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
    ) -> BasicObserverState:

    del agent_state

    timestep_buffer = add_to_buffer(
      buffer=self.buffer,
      buffer_state=observer_state.timestep_buffer,
      x=first_timestep,
    )

    observer_state = observer_state.replace(
      timestep_buffer=timestep_buffer,
      episodes=observer_state.episodes + 1,
      env_steps=observer_state.env_steps + 1,
    )

    return observer_state

  def observe(
    self,
    observer_state: BasicObserverState,
    predictions: struct.PyTreeNode,
    action: jax.Array,
    next_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
    ) -> None:
    """Update log and flush if terminal + log period hit.


    Args:
        agent_state (struct.PyTreeNode): _description_
        predictions (struct.PyTreeNode): _description_
        action (jax.Array): _description_
        next_timestep (TimeStep): _description_
    """
    del agent_state

    # update buffer with information
    timestep_buffer = add_to_buffer(
      buffer=self.buffer,
      buffer_state=observer_state.timestep_buffer,
      x=next_timestep,
    )

    prediction_buffer = add_to_buffer(
      buffer=self.buffer,
      buffer_state=observer_state.prediction_buffer,
      x=predictions,
    )

    action_buffer = add_to_buffer(
      buffer=self.buffer,
      buffer_state=observer_state.action_buffer,
      x=action,
    )

    # update return/length information
    episode_idx = observer_state.episodes
    episode_returns = observer_state.episode_returns.at[:, episode_idx].add(next_timestep.reward)
    episode_lengths = observer_state.episode_lengths.at[:, episode_idx].add(1)

    # update observer state
    observer_state = observer_state.replace(
      timestep_buffer=timestep_buffer,
      prediction_buffer=prediction_buffer,
      action_buffer=action_buffer,
      episode_returns=episode_returns,
      episode_lengths=episode_lengths,
      env_steps=observer_state.env_steps + 1,
    )
    
    #############
    # flushing and logging
    # it's easier to just do 1 env, so will ignore rest...
    # otherwise, having competing conditons when when episode ends
    #############
    # first_obs_state = jax.tree_map(lambda x: x[0], observer_state)
    first_next_timestep = jax.tree_map(lambda x: x[0], next_timestep)

    #-----------------------
    # if final time-step and log-period has been hit, flush the metrics
    #-----------------------
    flush = jnp.logical_and(
      first_next_timestep.last(),
      observer_state.episodes[0] % self.log_period == 0,
      )

    jax.lax.cond(
        flush,
        lambda: self.flush_metrics(observer_state),
        lambda: None,
    )

    #-----------------------
    # if final time-step, reset buffers. only keep around 1.
    #-----------------------
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
      episodes=observer_state.episodes + 1,
    )

  def flush_metrics(self, observer_state: BasicObserverState):
    def callback(os: BasicObserverState):
        if wandb.run is not None:
          metrics = {
            'avg_episode_length': os.episode_lengths.mean(),
            'avg_episode_return': os.episode_returns.mean(),
            'num_episodes': os.episodes.sum(),
            'num_steps': os.env_steps.sum(),
          }
          wandb.log(metrics)
    jax.debug.callback(callback, observer_state)

