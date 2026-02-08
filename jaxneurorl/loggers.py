from typing import Callable, Any, Optional


from flax import struct
from flax.training.train_state import TrainState
from gymnax.environments import environment
import jax
import jax.numpy as jnp
import tree
import wandb
import optax

from jaxneurorl.observers import BasicObserverState


def default_gradient_logger(
  train_state: TrainState, gradients: dict, key: str = "gradients"
):
  if "params" in gradients:
    gradients = gradients["params"]
  gradient_metrics = {
    f"{key}/{k}_norm": optax.global_norm(v) for k, v in gradients.items()
  }

  def callback(ts, g):
    if wandb.run is not None:
      g.update(
        {
          f"{key}/num_actor_steps": ts.timesteps,
          f"{key}/num_learner_updates": ts.n_updates,
        }
      )
      wandb.log(g)

  jax.debug.callback(callback, train_state, gradient_metrics)


def default_learner_logger(
  train_state: TrainState, learner_metrics: dict, key: str = "learner"
):
  def callback(ts: train_state, metrics: dict):
    metrics = {f"{key}/{k}": v for k, v in metrics.items()}
    metrics.update(
      {
        f"{key}/num_actor_steps": ts.timesteps,
        f"{key}/num_learner_updates": ts.n_updates,
      }
    )
    if wandb.run is not None:
      wandb.log(metrics)

  # prepare
  learner_metrics = jax.tree_util.tree_map(lambda x: x.mean(), learner_metrics)  # []
  jax.debug.callback(callback, train_state, learner_metrics)


def default_experience_logger(
  train_state: TrainState,
  observer_state: BasicObserverState,
  key: str = "train",
  log_details_period: int = 0,
):
  def callback(ts: train_state, os: BasicObserverState):
    # main
    end = min(os.idx + 1, len(os.episode_lengths))
    metrics = {
      f"{key}/avg_episode_length": os.episode_lengths[:end].mean(),
      f"{key}/avg_episode_return": os.episode_returns[:end].mean(),
      f"{key}/num_actor_steps": ts.timesteps,
      f"{key}/num_learner_updates": ts.n_updates,
    }
    if wandb.run is not None:
      wandb.log(metrics)

  jax.debug.callback(callback, train_state, observer_state)


@struct.dataclass
class Logger:
  gradient_logger: Callable[[TrainState, dict, str], Any]

  learner_logger: Callable[[TrainState, dict, str], Any]

  experience_logger: Callable[[TrainState, BasicObserverState, str], Any]

  learner_log_extra: Optional[Callable[[Any], Any]] = None


def default_make_logger(
  config: dict, env: environment.Environment, env_params: environment.EnvParams
):
  return Logger(
    gradient_logger=default_gradient_logger,
    learner_logger=default_learner_logger,
    experience_logger=default_experience_logger,
  )
