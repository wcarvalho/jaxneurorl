
import abc
import collections
from typing import Dict, Union, Optional, Callable, Optional

import jax
import jax.numpy as jnp
import flashbax as fbx
from flax import struct
from flax.training.train_state import TrainState
import numpy as np
import wandb
import matplotlib.pyplot as plt


from projects.humansf import keyroom
from projects.humansf.visualizer import plot_frames
from jaxneurorl.agents.basics import TimeStep

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


@struct.dataclass
class BasicObserverState:
  episode_returns: jax.Array
  episode_lengths: jax.Array
  finished: jax.Array
  action_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  timestep_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  prediction_buffer: fbx.trajectory_buffer.TrajectoryBufferState
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
      **kwargs,
      ):

    assert extract_task_info is not None

    self.extract_task_info = extract_task_info

    self.log_period = log_period
    self.max_episode_length = max_episode_length
    self.buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=max_episode_length*max_num_episodes,
        min_length_time_axis=1,
        sample_batch_size=1,  # unused
        add_batch_size=1,
        sample_sequence_length=1,  # unused
        period=1,
    )
    self.task_info_buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=max_num_episodes*10,
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
      episode_lengths=jnp.zeros((self.log_period), dtype=jnp.int32),
      finished=jnp.zeros((self.log_period), dtype=jnp.int32),
      task_info_buffer=self.task_info_buffer.init(
         get_first(self.extract_task_info(example_timestep))),
      timestep_buffer=self.buffer.init(get_first(example_timestep)),
      action_buffer=self.buffer.init(get_first(example_action)),
      prediction_buffer=self.buffer.init(get_first(example_predictions)),
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
      task_info_buffer=task_info_buffer,
      timestep_buffer=add_first_to_buffer(
        self.buffer, observer_state.timestep_buffer, first_timestep),
      )

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

    def advance_episode(os):
      # beginning new episode
      next_idx = os.idx + 1
      task_info_buffer = add_first_to_buffer(
            buffer=self.buffer,
            buffer_state=os.task_info_buffer,
            x=self.extract_task_info(next_timestep),
          )

      return os.replace(
        idx=next_idx,
        task_info_buffer=task_info_buffer,
        finished=os.finished.at[os.idx].add(1),
        episode_lengths=os.episode_lengths.at[next_idx].add(1),
        episode_returns=os.episode_returns.at[next_idx].add(first_next_timestep.reward),
        timestep_buffer=add_first_to_buffer(self.buffer, os.timestep_buffer, next_timestep),
        action_buffer=add_first_to_buffer(self.buffer, os.action_buffer, action),
        prediction_buffer=add_first_to_buffer(self.buffer, os.prediction_buffer, predictions),
        )

    def update_episode(os):
      # within same episode
      # update return/length information
      idx = os.idx
      # update observer state
      return os.replace(
        episode_lengths=os.episode_lengths.at[idx].add(1),
        episode_returns=os.episode_returns.at[idx].add(first_next_timestep.reward),
        timestep_buffer=add_first_to_buffer(self.buffer, os.timestep_buffer, next_timestep),
        action_buffer=add_first_to_buffer(self.buffer, os.action_buffer, action),
        prediction_buffer=add_first_to_buffer(self.buffer, os.prediction_buffer, predictions),
      )

    observer_state = jax.lax.cond(
      first_next_timestep.first(),
      advance_episode,
      update_episode,
      observer_state
    )

    return observer_state

def experience_logger(
        train_state: TrainState,
        observer_state: BasicObserverState,
        key: str = 'train',
        render_fn: Callable = None,
        log_details_period: int = 0,
        action_names: Optional[dict] = None,
        extract_task_info: Callable[[TimeStep], struct.PyTreeNode] = lambda t: t,
        get_task_name: Callable = lambda t: 'Task',
        max_len: int = 40,
        ):

    def callback(ts: TrainState, os: BasicObserverState):
        # main
        task_info_buffer = os.task_info_buffer.experience
        len_task_info = max((jax.tree_map(lambda x: x.shape[-1], task_info_buffer)).values())
        end = min(os.idx + 1, len(os.episode_lengths), len_task_info)

        #--------------------
        # per-task logging
        #--------------------
        metrics = collections.defaultdict(list)

        return_key = lambda name: f'{key}/0.1 {name.capitalize()} - AvgReturn'
        length_key = lambda name: f'{key}/1.1 {name.capitalize()} - AvgLength'
  
        for idx in range(end):
          task_info = jax.tree_map(lambda x: x[0, idx], os.task_info_buffer.experience)
          task_name = get_task_name(task_info)

          if os.finished[idx] > 0:
            metrics[return_key(task_name)].append(os.episode_returns[idx])
            metrics[length_key(task_name)].append(os.episode_lengths[idx])

        metrics = {k: np.array(v).mean() for k, v in metrics.items()}
        metrics.update({
            f'{key}/z. avg_episode_length': os.episode_lengths[:end].mean(),
            f'{key}/0.0 avg_episode_return': os.episode_returns[:end].mean(),
            f'{key}/num_actor_steps': ts.timesteps,
            f'{key}/num_learner_updates': ts.n_updates,
        })
        if wandb.run is not None:
          wandb.log(metrics)

        if log_details_period and (int(ts.n_logs) % int(log_details_period) == 0):
          timesteps = jax.tree_map(lambda x: x[0], os.timestep_buffer.experience)
          actions = jax.tree_map(lambda x: x[0], os.action_buffer.experience)
          #predictions = jax.tree_map(lambda x: x[0], os.prediction_buffer.experience)

          #################
          # frames
          #################
          obs_images = []
          for idx in range(max_len):
              index = lambda y: jax.tree_map(lambda x: x[idx], y)
              obs_image = render_fn(index(timesteps.state))
              obs_images.append(obs_image)

          #################
          # actions
          #################
          def action_name(a):
            if action_names is not None:
              name = action_names.get(int(a), 'ERROR?')
              return f"action {int(a)}: {name}"
            else:
              return f"action: {int(a)}"
          actions_taken = [action_name(a) for a in actions]

          #################
          # plot
          #################
          index = lambda t, idx: jax.tree_map(lambda x: x[idx], t)
          def panel_title_fn(timesteps, i):
            task_name = get_task_name(extract_task_info(index(timesteps, i)))
            title = f'{task_name}'

            step_type = int(timesteps.step_type[i])
            step_type = ['first', 'mid', '|last|'][step_type]
            title += f'\nt={i}, type={step_type}'

            if i < len(actions_taken):
              title += f'\n{actions_taken[i]}'
            title += f'\nr={timesteps.reward[i]}, $\\gamma={timesteps.discount[i]}$'

            return title

          fig = plot_frames(
              timesteps=timesteps,
              frames=obs_images,
              panel_title_fn=panel_title_fn,
              ncols=6)

          if wandb.run is not None:
              wandb.log({f"{key}_example/trajectory": wandb.Image(fig)})
          plt.close(fig)

    jax.debug.callback(callback, train_state, observer_state)


